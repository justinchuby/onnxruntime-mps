//! Env-gated observability tracing for the MLX EP — the "available slice" of the
//! Metal-tracing design (`docs/METAL_TRACING.md`) that our MLX-native architecture
//! actually permits.
//!
//! ## Why this shape (and not the design's per-kernel GPU timing)
//!
//! `mlx-c` exposes only `mlx_metal_is_available` + `mlx_metal_start_capture` /
//! `stop_capture` (the Xcode GPU-debugger capture). MLX creates, commits, and hides
//! its own `MTLCommandBuffer`s *inside* `mlx_eval`, and it fuses a whole subgraph
//! into ONE lazy graph → ONE eval. So the design's per-op `MTLCommandBuffer`
//! `gpuStartTime` (§4) and per-kernel `MTLCounterSampleBuffer` counters (§6) are
//! simply **not reachable** through `mlx-c` on the default fast path. What *is*
//! reachable, and what this module delivers:
//!
//!   * **Perfetto spans** (via `onnx-runtime-tracer`): one span per fused subgraph
//!     (`mlx.subgraph`, cat `ep`), a nested span around the **synchronous**
//!     `mlx_eval` (`mlx.eval`, cat `gpu`) whose CPU wall time is the *GPU-inclusive*
//!     time of the whole fused subgraph (that is the granularity MLX gives us on the
//!     fused path), a lightweight span per node at graph-build time (`<op_type>`,
//!     cat `op`), and a rich per-op detail span (shapes / dtype / elements / bytes).
//!   * **Seeing INSIDE the fused eval** — two opt-in modes break the single opaque
//!     `mlx.eval` blob into per-op detail (both keep the default path untouched):
//!       - **Fine per-op GPU timing** (`ONNX_GENAI_MLX_TRACE_FINE=1`): eval each
//!         node's outputs individually → a `gpu.op` span per op with GPU-inclusive
//!         time. BREAKS fusion (debug-only, slower). See [`MlxTracer::fine_enabled`].
//!       - **Xcode GPU capture** (`ONNX_GENAI_MLX_GPU_CAPTURE=<path.gputrace>`): wrap
//!         the first eval in `mlx_metal_start_capture`/`stop_capture` for a
//!         `.gputrace` bundle with full per-kernel timing. See
//!         [`MlxTracer::begin_gpu_capture`].
//!   * **os_signpost** intervals around the same subgraph / eval regions, so an
//!     Instruments *Metal System Trace* correlates. Zero cost when Instruments is
//!     not attached.
//!   * **GPU usage counters** (Chrome `"C"` phase, their own Perfetto tracks):
//!     `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`), `mlx.gpu_mem_pct`
//!     (allocated / `recommendedMaxWorkingSetSize`), and `mlx.gpu_util_pct` — GPU
//!     active-residency % via the private **IOReport** framework (the `GPUPH`
//!     "GPU Performance States" channel, resolved by `dlopen`/`dlsym`; see
//!     [`ioreport`]). Degrades to no counter if IOReport is unavailable.
//!   * **Slowest-ops summary** at teardown ([`MlxTracer::log_slowest_ops`]): a
//!     compact top-10 (op_type → total µs, %, calls) to stderr + trace metadata.
//!
//! Everything is gated on an atomic enable flag inside `TraceContext`: with tracing
//! OFF (env unset) every entry point is a single relaxed atomic load + early return,
//! so a production run pays essentially nothing (the design's "0% when off" rule) and
//! the single fused `mlx_eval` is left exactly as-is.
//!
//! All `unsafe` FFI (Metal/objc for the memory counter, the mlx-c Metal capture, the
//! IOReport GPU-util sampler, os_signpost for the intervals) is confined to this
//! module; the op/engine code stays clean.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use onnx_runtime_tracer::{Args, MemoryCollector, SpanGuard, TraceContext};
use std::sync::Arc;

/// Which execution path a node's handler took, declared via
/// [`TranslationContext::mark_fast`](crate::engine::TranslationContext::mark_fast) /
/// [`mark_composed`](crate::engine::TranslationContext::mark_composed).
///
/// * [`PathMark::Fast`] — the node used a fused MLX kernel (the intended fast path).
/// * [`PathMark::Composed`] — the node *has* a fused kernel available but instead fell
///   back to a slower composed / generic implementation. These stand out in the trace.
///
/// Ops with no fast/slow distinction (ordinary elementwise/shape ops) leave no mark and
/// are treated as neutral.
pub enum PathMark {
    /// Fused MLX kernel used (green/normal). Carries the kernel name.
    Fast(&'static str),
    /// Composed/fallback path taken despite a fused kernel existing (SLOW). Carries the reason.
    Composed(String),
}

/// Set to a filesystem path to enable tracing; the Chrome/Perfetto JSON trace is
/// written there on EP teardown. Unset → tracing disabled (near-zero cost).
pub const TRACE_ENV: &str = "ONNX_GENAI_MLX_TRACE";
/// Set to `1` to force os_signpost intervals on even when JSON tracing is off.
pub const SIGNPOST_ENV: &str = "ONNX_GENAI_MLX_SIGNPOST";
/// Set to `1` to enable **fine-grained per-op GPU timing** (implies JSON tracing on).
///
/// In this mode every node's output array(s) are `mlx_array_eval`'d individually right
/// after its handler binds them, so each op is materialized on its own and its
/// GPU-inclusive wall time is recorded as a distinct `gpu.op` span. This BREAKS MLX's
/// subgraph fusion (materializing per node defeats the lazy graph), so it is
/// **slower** and strictly a debug tool — the normal path (fine off) keeps the single
/// fused `mlx_eval`. See [`MlxTracer::fine_enabled`].
pub const FINE_ENV: &str = "ONNX_GENAI_MLX_TRACE_FINE";
/// Set to a `<path>.gputrace` (or `1` for a default path) to wrap the **first** boundary
/// eval in a Metal GPU capture (`mlx_metal_start_capture` … `stop_capture`). The
/// resulting `.gputrace` bundle opens in Xcode / Instruments for full per-kernel GPU
/// timing, occupancy and memory-bandwidth — the detail `mlx-c` cannot surface itself.
/// Only the first eval is captured (a whole-decode capture would be enormous).
pub const GPU_CAPTURE_ENV: &str = "ONNX_GENAI_MLX_GPU_CAPTURE";

/// Process-wide tracer singleton. All subgraphs/sessions share one timeline and one
/// output file, stamped with the real `pid` so the events merge into onnx-genai's
/// Perfetto timeline under the same process, on their own tracks.
static TRACER: OnceLock<MlxTracer> = OnceLock::new();

/// The shared tracer. First access reads the environment and wires everything up.
pub fn tracer() -> &'static MlxTracer {
    TRACER.get_or_init(MlxTracer::new)
}

thread_local! {
    static THREAD_NAMED: Cell<bool> = const { Cell::new(false) };
}

/// One sampled counter point (rendered as a Chrome `"C"` phase event at export).
struct CounterSample {
    track: String,
    key: String,
    value: f64,
    ts: u64,
}

/// The env-gated tracer. Cheap to leave wired in when disabled.
pub struct MlxTracer {
    ctx: TraceContext,
    mem: Option<Arc<MemoryCollector>>,
    path: Option<PathBuf>,
    counters: Mutex<Vec<CounterSample>>,
    /// Cumulative composed-path hit count per op-type (drives `mlx.composed_path_count`).
    composed_counts: Mutex<HashMap<String, u64>>,
    /// `os_log_t` for signposts as a `usize` (0 = disabled) so the struct is `Send + Sync`.
    signpost_log: usize,
    /// Cached default `MTLDevice` as a `usize` (0 = unavailable).
    device: usize,
    /// Fine-grained per-op GPU timing mode (`ONNX_GENAI_MLX_TRACE_FINE`). Breaks fusion.
    fine: bool,
    /// Resolved `.gputrace` capture path (`None` = capture disabled).
    capture_path: Option<PathBuf>,
    /// Guards the one-shot Metal capture so only the FIRST eval is captured.
    capture_done: AtomicBool,
    /// Cumulative per-op-type time for the end-of-run "slowest ops" summary:
    /// `op_type -> (total_us, call_count)`. Populated with GPU-inclusive times in fine
    /// mode, otherwise with build/handler wall times.
    op_times: Mutex<HashMap<String, (u64, u64)>>,
    /// IOReport GPU-utilisation sampler (`None` when unavailable). See [`ioreport`].
    gpu_util: Mutex<Option<ioreport::GpuUtil>>,
}

// The stored pointers are only ever used through the confined FFI helpers below and
// never dereferenced as Rust references, so sharing them across threads is sound.
unsafe impl Send for MlxTracer {}
unsafe impl Sync for MlxTracer {}

impl MlxTracer {
    fn new() -> Self {
        let path = std::env::var(TRACE_ENV).ok().filter(|s| !s.is_empty());
        let fine = std::env::var(FINE_ENV).map(|v| v == "1").unwrap_or(false);
        // Fine mode implies JSON tracing on even if TRACE_ENV is unset (the spans still
        // accumulate in memory; export is a no-op without a path).
        let trace_on = path.is_some() || fine;

        let (ctx, mem) = if trace_on {
            let (ctx, mem) = TraceContext::in_memory();
            ctx.set_process_name("onnxruntime-mlx-ep");
            (ctx, Some(mem))
        } else {
            (TraceContext::noop(), None)
        };

        // GPU capture (`ONNX_GENAI_MLX_GPU_CAPTURE`) is independent of JSON tracing: it
        // writes a `.gputrace` bundle, not JSON. `1` → a default path in the cwd.
        let capture_path = std::env::var(GPU_CAPTURE_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| {
                if s == "1" {
                    PathBuf::from("mlx_capture.gputrace")
                } else {
                    PathBuf::from(s)
                }
            });

        let signpost_on = trace_on
            || std::env::var(SIGNPOST_ENV)
                .map(|v| v == "1")
                .unwrap_or(false);
        let signpost_log = if signpost_on {
            signpost::create_log()
        } else {
            0
        };

        // A default device is only needed for the GPU-memory counter, so create it
        // once (and leak it — one device handle for the process lifetime) when JSON
        // tracing is enabled.
        let device = if trace_on { gpu::default_device() } else { 0 };

        // Best-effort IOReport GPU-utilisation sampler (private framework, no sudo).
        let gpu_util = if trace_on { ioreport::GpuUtil::new() } else { None };

        MlxTracer {
            ctx,
            mem,
            path: path.map(PathBuf::from),
            counters: Mutex::new(Vec::new()),
            composed_counts: Mutex::new(HashMap::new()),
            signpost_log,
            device,
            fine,
            capture_path,
            capture_done: AtomicBool::new(false),
            op_times: Mutex::new(HashMap::new()),
            gpu_util: Mutex::new(gpu_util),
        }
    }

    /// Whether JSON tracing is enabled (the hot-path gate).
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.ctx.is_enabled()
    }

    /// Name the current OS thread's track once (idempotent per thread).
    pub fn note_thread(&self, name: &str) {
        if !self.is_enabled() {
            return;
        }
        THREAD_NAMED.with(|n| {
            if !n.get() {
                self.ctx.set_thread_name(name);
                n.set(true);
            }
        });
    }

    /// Span + signpost interval around one fused subgraph's whole Compute.
    pub fn subgraph_region(&self, node_count: usize) -> Region {
        let span = self
            .ctx
            .span("mlx.subgraph", "ep")
            .with_args(Args::new().with("nodes", node_count as u64));
        let sp = signpost::interval_begin(self.signpost_log, SP_SUBGRAPH);
        Region { _span: span, signpost: sp }
    }

    /// Span + signpost interval around the synchronous `mlx_eval` (GPU-inclusive time).
    pub fn eval_region(&self) -> Region {
        let span = self
            .ctx
            .span("mlx.eval", "gpu")
            .with_args(Args::new().device("gpu"));
        let sp = signpost::interval_begin(self.signpost_log, SP_EVAL);
        Region { _span: span, signpost: sp }
    }

    /// Lightweight build-time span for one node (records the op structure of a subgraph).
    pub fn op_span(&self, op_type: &str, num_inputs: usize, num_outputs: usize) -> SpanGuard {
        if !self.is_enabled() {
            return self.ctx.span(op_type, "op"); // inert guard, no allocation
        }
        self.ctx.span(op_type.to_string(), "op").with_args(
            Args::new()
                .with("op_type", op_type.to_string())
                .with("inputs", num_inputs as u64)
                .with("outputs", num_outputs as u64),
        )
    }

    /// Start a wall-clock timer for one node's handler, or `None` when tracing is off
    /// (the hot-path gate — no clock read when disabled).
    #[inline]
    pub fn op_timer_start(&self) -> Option<Instant> {
        if self.is_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Surface the fast-vs-composed path a node's handler declared (see [`PathMark`]).
    ///
    /// * `Fast(kernel)` → a `<Op> [fast]` span in cat `op.fast` with `optimized=true`
    ///   + `kernel=<...>`.
    /// * `Composed(reason)` → a `<Op> [composed]` span in the distinct cat `op.composed`
    ///   (Perfetto colours it differently) with `optimized=false` + `reason=<...>`, PLUS a
    ///   visible instant marker `⚠ composed-path: <Op> (<reason>)` on the timeline, PLUS a
    ///   bump of the per-op-type `mlx.composed_path_count` counter track.
    /// * `None` → neutral op, nothing emitted.
    ///
    /// No-op when tracing is disabled.
    pub fn record_op_path(&self, op_type: &str, start: Option<Instant>, mark: Option<PathMark>) {
        if !self.is_enabled() {
            return;
        }
        let Some(mark) = mark else {
            return;
        };
        match mark {
            PathMark::Fast(kernel) => {
                if let Some(start) = start {
                    self.ctx.complete(
                        format!("{op_type} [fast]"),
                        "op.fast",
                        start,
                        start.elapsed(),
                        Some(
                            Args::new()
                                .with("op_type", op_type.to_string())
                                .with("optimized", true)
                                .with("kernel", kernel),
                        ),
                    );
                }
            }
            PathMark::Composed(reason) => {
                if let Some(start) = start {
                    self.ctx.complete(
                        format!("{op_type} [composed]"),
                        "op.composed",
                        start,
                        start.elapsed(),
                        Some(
                            Args::new()
                                .with("op_type", op_type.to_string())
                                .with("optimized", false)
                                .with("reason", reason.clone()),
                        ),
                    );
                }
                // A standalone mark on the timeline so a composed path is impossible to miss.
                self.ctx.instant(
                    format!("⚠ composed-path: {op_type} ({reason})"),
                    "op.composed",
                    Some(
                        Args::new()
                            .with("op_type", op_type.to_string())
                            .with("optimized", false)
                            .with("reason", reason),
                    ),
                );
                self.bump_composed_counter(op_type);
            }
        }
    }

    /// Increment and emit the cumulative composed-path counter for `op_type` as a point on
    /// the `mlx.composed_path_count` Perfetto counter track (one series per op-type).
    fn bump_composed_counter(&self, op_type: &str) {
        let ts = self.ctx.clock().now_micros();
        let value = {
            let mut counts = match self.composed_counts.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            let n = counts.entry(op_type.to_string()).or_insert(0);
            *n += 1;
            *n as f64
        };
        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: "mlx.composed_path_count".to_string(),
            key: op_type.to_string(),
            value,
            ts,
        });
    }

    /// Whether **fine-grained per-op GPU timing** mode is on (`ONNX_GENAI_MLX_TRACE_FINE`).
    /// When true the engine eval's each node's outputs individually to time them — this
    /// BREAKS fusion and is a debug-only mode. Implies [`is_enabled`](Self::is_enabled).
    #[inline]
    pub fn fine_enabled(&self) -> bool {
        self.fine
    }

    /// Emit a rich per-op span (cat `op`) for a node whose outputs are already bound.
    /// Carries input/output shapes, dtype, element count and byte size so every op span
    /// has resource context even without fine mode. Also feeds the slowest-ops summary
    /// with the build/handler wall time. No-op when tracing is disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn record_op_meta(
        &self,
        op_type: &str,
        start: Instant,
        dur: Duration,
        out_shapes: &str,
        in_shapes: &str,
        dtype: &str,
        elements: u64,
        bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.ctx.complete(
            op_type.to_string(),
            "op",
            start,
            dur,
            Some(
                Args::new()
                    .with("op_type", op_type.to_string())
                    .with("output_shapes", out_shapes.to_string())
                    .with("input_shapes", in_shapes.to_string())
                    .with("dtype", dtype.to_string())
                    .with("elements", elements)
                    .with("bytes", bytes),
            ),
        );
        self.record_op_time(op_type, dur.as_micros() as u64);
    }

    /// Emit a per-op **GPU** span (cat `gpu.op`) for fine mode — its wall time is the
    /// GPU-INCLUSIVE time of just this op's forced eval (fusion broken). Carries the same
    /// resource Args and feeds the slowest-ops summary with GPU time. No-op when disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn record_gpu_op(
        &self,
        op_type: &str,
        start: Instant,
        dur: Duration,
        out_shapes: &str,
        in_shapes: &str,
        dtype: &str,
        elements: u64,
        bytes: u64,
    ) {
        if !self.is_enabled() {
            return;
        }
        self.ctx.complete(
            op_type.to_string(),
            "gpu.op",
            start,
            dur,
            Some(
                Args::new()
                    .device("gpu")
                    .with("op_type", op_type.to_string())
                    .with("output_shapes", out_shapes.to_string())
                    .with("input_shapes", in_shapes.to_string())
                    .with("dtype", dtype.to_string())
                    .with("elements", elements)
                    .with("bytes", bytes),
            ),
        );
        self.record_op_time(op_type, dur.as_micros() as u64);
    }

    /// Accumulate one op-type timing sample for the end-of-run slowest-ops summary.
    fn record_op_time(&self, op_type: &str, us: u64) {
        let mut m = match self.op_times.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let e = m.entry(op_type.to_string()).or_insert((0, 0));
        e.0 += us;
        e.1 += 1;
    }

    /// Log a compact **top-10 slowest ops** summary (op_type → total us, %, call count)
    /// to stderr AND as a `mlx.slowest_ops` trace-metadata instant, so an agent can see
    /// e.g. "MatMul = 62% of GPU time" without parsing the whole JSON. In fine mode the
    /// times are GPU-inclusive; otherwise they are build/handler wall times (noted).
    /// No-op when tracing is disabled or no ops were recorded.
    pub fn log_slowest_ops(&self) {
        if !self.is_enabled() {
            return;
        }
        let snapshot: Vec<(String, u64, u64)> = {
            let m = match self.op_times.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            m.iter().map(|(k, v)| (k.clone(), v.0, v.1)).collect()
        };
        if snapshot.is_empty() {
            return;
        }
        let total: u64 = snapshot.iter().map(|(_, us, _)| *us).sum();
        let mut ranked = snapshot;
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(10);

        let kind = if self.fine {
            "GPU-inclusive per-op"
        } else {
            "build-time (fusion intact; per-op GPU time needs ONNX_GENAI_MLX_TRACE_FINE=1)"
        };
        let denom = total.max(1) as f64;

        let mut lines = String::new();
        lines.push_str(&format!(
            "[rust-mlx-ep] slowest ops ({kind}), total {total} us across {} op-type(s):\n",
            ranked.len()
        ));
        let mut args = Args::new().with("timing_kind", kind).with("total_us", total);
        for (i, (op, us, calls)) in ranked.iter().enumerate() {
            let pct = (*us as f64 / denom) * 100.0;
            lines.push_str(&format!(
                "  {:>2}. {:<20} {:>10} us  {:>5.1}%  ({} call(s))\n",
                i + 1,
                op,
                us,
                pct,
                calls
            ));
            args = args.with(
                format!("{:02}_{op}", i + 1),
                format!("{us}us {pct:.1}% x{calls}"),
            );
        }
        eprint!("{lines}");
        self.ctx.instant("mlx.slowest_ops", "summary", Some(args));
    }

    /// Begin the one-shot Metal GPU capture around the FIRST eval, returning a guard that
    /// stops the capture (and logs the written path) on drop. Returns `None` when capture
    /// is disabled, already taken, or Metal is unavailable. Independent of JSON tracing.
    #[must_use]
    pub fn begin_gpu_capture(&self) -> Option<CaptureGuard> {
        let path = self.capture_path.as_ref()?;
        // Take the one-shot slot; only the first eval wins.
        if self
            .capture_done
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }
        if !metal_capture::is_available() {
            eprintln!(
                "[rust-mlx-ep] GPU capture requested but Metal is unavailable (mlx_metal_is_available=false); skipping"
            );
            return None;
        }
        // The Metal capture layer must be inserted BEFORE the process creates its
        // MTLDevice, which only happens when `MTL_CAPTURE_ENABLED=1` is exported in the
        // environment. Without it `mlx_metal_start_capture` hits MLX's fatal error
        // handler (which aborts the process), so we refuse up-front with a clear message
        // rather than crash the run.
        let capture_layer_on = std::env::var("MTL_CAPTURE_ENABLED")
            .map(|v| v == "1")
            .unwrap_or(false);
        if !capture_layer_on {
            eprintln!(
                "[rust-mlx-ep] GPU capture requires MTL_CAPTURE_ENABLED=1 to be exported before the \
                 process starts (the Metal capture layer must be inserted at device creation); \
                 skipping capture. Re-run with: MTL_CAPTURE_ENABLED=1 ONNX_GENAI_MLX_GPU_CAPTURE={} ...",
                path.to_string_lossy()
            );
            return None;
        }
        let path_str = path.to_string_lossy().to_string();
        if metal_capture::start(&path_str) {
            eprintln!("[rust-mlx-ep] Metal GPU capture STARTED → {path_str} (first eval only)");
            Some(CaptureGuard { path: path_str })
        } else {
            eprintln!(
                "[rust-mlx-ep] GPU capture start FAILED for {path_str} \
                 (capture requires MTL_CAPTURE_ENABLED=1 in the environment and a path ending in .gputrace)"
            );
            None
        }
    }


    /// Sample GPU usage counters (cheap; only when tracing is enabled).
    ///
    /// Emits `mlx.gpu_mem_bytes` (`MTLDevice.currentAllocatedSize`) and, when the
    /// device reports a working-set budget, `mlx.gpu_mem_pct`. When the IOReport GPU
    /// sampler initialised, also emits `mlx.gpu_util_pct` (GPU active-residency %) and,
    /// when available, `mlx.gpu_freq_mhz` — the utilisation signal `macmon`/`asitop`/
    /// Activity Monitor read from the private IOReport framework (no sudo). If IOReport
    /// was unavailable the util counters are simply skipped (see [`ioreport`]).
    pub fn sample_gpu_counters(&self) {
        if !self.is_enabled() || self.device == 0 {
            return;
        }
        let dev = self.device;
        let allocated = gpu::msg_u64(dev, b"currentAllocatedSize\0") as f64;
        let recommended = gpu::msg_u64(dev, b"recommendedMaxWorkingSetSize\0") as f64;
        let ts = self.ctx.clock().now_micros();

        let mut c = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        c.push(CounterSample {
            track: "mlx.gpu_mem_bytes".to_string(),
            key: "bytes".to_string(),
            value: allocated,
            ts,
        });
        if recommended > 0.0 {
            c.push(CounterSample {
                track: "mlx.gpu_mem_pct".to_string(),
                key: "pct".to_string(),
                value: (allocated / recommended) * 100.0,
                ts,
            });
        }
        drop(c);

        // GPU utilisation % (and freq) via IOReport — a real delta between this sample
        // and the previous one. First call primes the baseline and yields nothing.
        if let Ok(mut util) = self.gpu_util.lock() {
            if let Some(sampler) = util.as_mut() {
                if let Some(reading) = sampler.sample() {
                    let mut c = match self.counters.lock() {
                        Ok(g) => g,
                        Err(p) => p.into_inner(),
                    };
                    c.push(CounterSample {
                        track: "mlx.gpu_util_pct".to_string(),
                        key: "pct".to_string(),
                        value: reading.active_pct,
                        ts,
                    });
                    if let Some(mhz) = reading.freq_mhz {
                        c.push(CounterSample {
                            track: "mlx.gpu_freq_mhz".to_string(),
                            key: "mhz".to_string(),
                            value: mhz,
                            ts,
                        });
                    }
                }
            }
        }
    }

    /// Write the accumulated trace (spans + counter events) as a Chrome Trace JSON
    /// array to the configured path. No-op when tracing is disabled.
    ///
    /// Called on every EP teardown. The `MemoryCollector` accumulates events across
    /// all sessions in the process, so each call rewrites the **full cumulative**
    /// trace (last writer wins / write-once semantics); the final teardown leaves the
    /// complete trace on disk.
    pub fn export(&self) {
        if !self.is_enabled() {
            return;
        }
        let (Some(mem), Some(path)) = (&self.mem, &self.path) else {
            return;
        };

        // Base document from the tracer: a Chrome Trace JSON array "[ {..}, .. ]".
        let mut out = mem.to_chrome_json();

        // Build the counter events ("C" phase) manually — the tracer's TracePhase has
        // no Counter variant — and splice them into the same array.
        let pid = self.ctx.pid();
        let counters = match self.counters.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut tail = String::new();
        for c in counters.iter() {
            tail.push_str(&format!(
                ",{{\"name\":\"{}\",\"cat\":\"gpu_counter\",\"ph\":\"C\",\"ts\":{},\
                 \"pid\":{},\"tid\":0,\"args\":{{\"{}\":{}}}}}",
                c.track, c.ts, pid, c.key, c.value
            ));
        }

        // Splice `tail` (each element prefixed with a comma) before the closing ']'.
        if out.ends_with(']') {
            out.pop();
            let had_events = out.len() > 1; // more than just "["
            if !tail.is_empty() {
                if had_events {
                    out.push_str(&tail);
                } else {
                    out.push_str(&tail[1..]); // strip the leading comma
                }
            }
            out.push(']');
        }

        match std::fs::write(path, &out) {
            Ok(()) => eprintln!(
                "[rust-mlx-ep] wrote MLX trace ({} span event(s), {} counter sample(s)) to {}",
                mem.len(),
                counters.len(),
                path.display()
            ),
            Err(e) => eprintln!("[rust-mlx-ep] trace export to {} failed: {e}", path.display()),
        }
    }
}

/// RAII cover for a traced region: an `onnx-runtime-tracer` span plus an optional
/// os_signpost interval. Both close (record / emit END) when the `Region` drops.
#[must_use = "a Region records its span/interval only while alive; drop it at the end of the region"]
pub struct Region {
    _span: SpanGuard,
    signpost: Option<signpost::Interval>,
}

impl Drop for Region {
    fn drop(&mut self) {
        if let Some(iv) = self.signpost.take() {
            iv.end();
        }
        // `_span` records on its own Drop.
    }
}

/// RAII guard for the one-shot Metal GPU capture: stops the capture (writing the
/// `.gputrace` bundle) and logs the path when dropped. Created by
/// [`MlxTracer::begin_gpu_capture`].
#[must_use = "the GPU capture only covers the region this guard is alive for"]
pub struct CaptureGuard {
    path: String,
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        metal_capture::stop();
        eprintln!(
            "[rust-mlx-ep] Metal GPU capture STOPPED → wrote {} (open in Xcode: `open {}`)",
            self.path, self.path
        );
    }
}

// Static signpost interval names (must outlive the interval; os_signpost takes a
// `const char *`).
const SP_SUBGRAPH: &[u8] = b"mlx.subgraph\0";
const SP_EVAL: &[u8] = b"mlx.eval\0";

// ---------------------------------------------------------------------------
// Confined FFI: Metal/objc for the GPU-memory counter.
// ---------------------------------------------------------------------------

mod gpu {
    use std::os::raw::{c_char, c_void};

    #[allow(non_camel_case_types)]
    type SEL = *const c_void;

    unsafe extern "C" {
        fn MTLCreateSystemDefaultDevice() -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> SEL;
        fn objc_msgSend();
    }

    /// The system default `MTLDevice` as a `usize` (0 if there is no GPU). Leaked on
    /// purpose: one device handle lives for the process.
    pub fn default_device() -> usize {
        unsafe { MTLCreateSystemDefaultDevice() as usize }
    }

    /// Send a nullary Objective-C message returning an unsigned integer
    /// (`NSUInteger` / `uint64_t`) — used for `currentAllocatedSize` and
    /// `recommendedMaxWorkingSetSize`. `sel_name` must be nul-terminated.
    pub fn msg_u64(obj: usize, sel_name: &[u8]) -> u64 {
        if obj == 0 {
            return 0;
        }
        unsafe {
            let sel = sel_registerName(sel_name.as_ptr() as *const c_char);
            // objc_msgSend is variadic/untyped in the header; transmute to the exact
            // shape of the message we are sending. On arm64 the integer result comes
            // back in x0 for this signature.
            let send: extern "C" fn(*mut c_void, SEL) -> u64 =
                std::mem::transmute(objc_msgSend as *const c_void);
            send(obj as *mut c_void, sel)
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: os_signpost intervals (Apple's ITT equivalent, design §5).
// Zero cost when Instruments is not recording.
// ---------------------------------------------------------------------------

mod signpost {
    use std::os::raw::{c_char, c_void};

    type OsLog = *mut c_void;

    // os_signpost_type_t values from <os/signpost.h>.
    const OS_SIGNPOST_INTERVAL_BEGIN: u8 = 1;
    const OS_SIGNPOST_INTERVAL_END: u8 = 2;

    unsafe extern "C" {
        fn os_log_create(subsystem: *const c_char, category: *const c_char) -> OsLog;
        fn os_signpost_id_generate(log: OsLog) -> u64;
        fn _os_signpost_emit_with_name_impl(
            dso: *mut c_void,
            log: OsLog,
            ty: u8,
            spid: u64,
            name: *const c_char,
            format: *const c_char,
            buf: *mut u8,
            size: u32,
        );
        // Per-image handle the os_signpost macros pass so Instruments can attribute
        // the emit to this dylib. Provided by the linker for every Mach-O image.
        static __dso_handle: c_void;
    }

    /// Create the signpost log, returning its `os_log_t` as a `usize` (0 on failure).
    pub fn create_log() -> usize {
        let subsystem = b"com.onnxruntime.mlx\0";
        let category = b"MLXExecutionProvider\0";
        unsafe {
            os_log_create(
                subsystem.as_ptr() as *const c_char,
                category.as_ptr() as *const c_char,
            ) as usize
        }
    }

    /// An open signpost interval; call [`Interval::end`] (done by `Region`'s Drop).
    pub struct Interval {
        log: usize,
        id: u64,
        name: *const c_char,
    }

    impl Interval {
        pub fn end(self) {
            // os_log expects a valid (even if empty) encoded arg buffer; a 2-byte
            // zeroed header (summary flags = 0, arg count = 0) is the no-args form.
            let mut buf: [u8; 2] = [0, 0];
            unsafe {
                _os_signpost_emit_with_name_impl(
                    &__dso_handle as *const c_void as *mut c_void,
                    self.log as OsLog,
                    OS_SIGNPOST_INTERVAL_END,
                    self.id,
                    self.name,
                    b"\0".as_ptr() as *const c_char,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                );
            }
        }
    }

    /// Begin an interval on `log` (a `usize` os_log_t) with the static `name`.
    /// Returns `None` when signposts are disabled (`log == 0`).
    pub fn interval_begin(log: usize, name: &'static [u8]) -> Option<Interval> {
        if log == 0 {
            return None;
        }
        let name_ptr = name.as_ptr() as *const c_char;
        let mut buf: [u8; 2] = [0, 0];
        unsafe {
            let id = os_signpost_id_generate(log as OsLog);
            _os_signpost_emit_with_name_impl(
                &__dso_handle as *const c_void as *mut c_void,
                log as OsLog,
                OS_SIGNPOST_INTERVAL_BEGIN,
                id,
                name_ptr,
                b"\0".as_ptr() as *const c_char,
                buf.as_mut_ptr(),
                buf.len() as u32,
            );
            Some(Interval { log, id, name: name_ptr })
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: Metal GPU capture via mlx-c (the Xcode GPU-debugger capture).
//
// mlx-c exposes exactly three entry points for this; MLX drives the underlying
// MTLCaptureManager itself, so wrapping the FIRST eval in start/stop produces a
// `.gputrace` bundle with full per-kernel GPU timing / occupancy / bandwidth — the
// detail mlx-c will not surface programmatically.
// ---------------------------------------------------------------------------

mod metal_capture {
    use crate::sys::mlx;
    use std::ffi::CString;

    /// `mlx_metal_is_available()` — false on machines without a Metal GPU.
    pub fn is_available() -> bool {
        let mut res = false;
        // Returns non-zero on error; treat any error as "unavailable".
        let rc = unsafe { mlx::mlx_metal_is_available(&mut res as *mut bool) };
        rc == 0 && res
    }

    /// Start a capture writing to `path` (must end in `.gputrace`). Returns whether it started.
    pub fn start(path: &str) -> bool {
        let Ok(c) = CString::new(path) else {
            return false;
        };
        let rc = unsafe { mlx::mlx_metal_start_capture(c.as_ptr()) };
        rc == 0
    }

    /// Stop the in-flight capture (flushes the `.gputrace` bundle to disk).
    pub fn stop() {
        unsafe {
            mlx::mlx_metal_stop_capture();
        }
    }
}

// ---------------------------------------------------------------------------
// Confined FFI: GPU utilisation % via the private IOReport framework.
//
// This is the signal `macmon`/`asitop`/Activity Monitor read (no sudo). We resolve
// the IOReport + CoreFoundation symbols with `dlopen`/`dlsym` at runtime so there is
// NO link-time dependency on a private framework — if IOReport is missing or its ABI
// differs, the sampler simply reports itself unavailable and no util counter is
// emitted (graceful degradation; never a crash on the traced-off fast path).
//
// GPU active-residency comes from the "GPU Stats" group, "GPU Performance States"
// channel: a state-residency channel whose per-state residencies we delta between two
// samples. active% = 100 * (residency in non-idle states) / (residency in all states).
// ---------------------------------------------------------------------------

mod ioreport {
    use std::os::raw::{c_char, c_int, c_longlong, c_void};

    type CFTypeRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CFMutableDictionaryRef = *mut c_void;
    type CFStringRef = *const c_void;
    type IOReportSubscriptionRef = *const c_void;
    // An IOReport "sample" channel handle passed to the iterate block.
    type IOReportSampleRef = *const c_void;

    const RTLD_NOW: c_int = 2;
    // kCFStringEncodingUTF8
    const CF_UTF8: u32 = 0x0800_0100;
    // IOReportIterate block return code to continue iterating.
    const K_IO_REPORT_ITER_OK: c_int = 0;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    // CoreFoundation functions we need, resolved via dlsym (CF is always present, but
    // resolving dynamically keeps this module self-contained / link-free).
    type CFStringCreateWithCStringFn =
        unsafe extern "C" fn(CFTypeRef, *const c_char, u32) -> CFStringRef;
    type CFStringGetCStringFn =
        unsafe extern "C" fn(CFStringRef, *mut c_char, c_longlong, u32) -> bool;
    type CFReleaseFn = unsafe extern "C" fn(CFTypeRef);

    // IOReport private functions.
    type IOReportCopyChannelsInGroupFn = unsafe extern "C" fn(
        CFStringRef,
        CFStringRef,
        u64,
        u64,
        u64,
    ) -> CFMutableDictionaryRef;
    type IOReportCreateSubscriptionFn = unsafe extern "C" fn(
        *mut c_void,
        CFMutableDictionaryRef,
        *mut CFMutableDictionaryRef,
        u64,
        CFTypeRef,
    ) -> IOReportSubscriptionRef;
    type IOReportCreateSamplesFn = unsafe extern "C" fn(
        IOReportSubscriptionRef,
        CFMutableDictionaryRef,
        CFTypeRef,
    ) -> CFDictionaryRef;
    type IOReportCreateSamplesDeltaFn =
        unsafe extern "C" fn(CFDictionaryRef, CFDictionaryRef, CFTypeRef) -> CFDictionaryRef;
    // IOReportIterate takes an Objective-C block; we pass a no-capture global block.
    type IOReportIterateFn = unsafe extern "C" fn(CFDictionaryRef, *const c_void);
    type IOReportChannelGetGroupFn = unsafe extern "C" fn(IOReportSampleRef) -> CFStringRef;
    type IOReportChannelGetChannelNameFn =
        unsafe extern "C" fn(IOReportSampleRef) -> CFStringRef;
    type IOReportStateGetCountFn = unsafe extern "C" fn(IOReportSampleRef) -> c_int;
    type IOReportStateGetNameForIndexFn =
        unsafe extern "C" fn(IOReportSampleRef, c_int) -> CFStringRef;
    type IOReportStateGetResidencyFn =
        unsafe extern "C" fn(IOReportSampleRef, c_int) -> c_longlong;

    /// One GPU-utilisation reading (active-residency %; freq is best-effort/None here).
    pub struct Reading {
        pub active_pct: f64,
        pub freq_mhz: Option<f64>,
    }

    /// The resolved IOReport symbol table + a live subscription and the previous sample.
    pub struct GpuUtil {
        sub: IOReportSubscriptionRef,
        channels: CFMutableDictionaryRef,
        prev: CFDictionaryRef,
        cf_release: CFReleaseFn,
        create_samples: IOReportCreateSamplesFn,
        create_delta: IOReportCreateSamplesDeltaFn,
        iterate: IOReportIterateFn,
    }

    // The subscription/channel handles live for the process; only touched under the
    // tracer's mutex, so sharing across threads is sound.
    unsafe impl Send for GpuUtil {}

    // Thread-local accumulator the no-capture iterate block writes into. IOReportIterate
    // is called synchronously on the calling thread, so a thread-local is safe.
    thread_local! {
        static ACC: std::cell::Cell<(i64, i64)> = const { std::cell::Cell::new((0, 0)) };
    }

    // --- Block ABI (a no-capture global block; invoke is a plain fn pointer) ---
    #[repr(C)]
    struct BlockDescriptor {
        reserved: u64,
        size: u64,
    }
    #[repr(C)]
    struct Block {
        isa: *const c_void,
        flags: c_int,
        reserved: c_int,
        invoke: extern "C" fn(*mut Block, IOReportSampleRef) -> c_int,
        descriptor: *const BlockDescriptor,
    }
    unsafe impl Sync for Block {}

    unsafe extern "C" {
        // The global-block "isa" the Objective-C runtime uses for stateless blocks.
        static _NSConcreteGlobalBlock: [*const c_void; 32];
    }

    static BLOCK_DESCRIPTOR: BlockDescriptor = BlockDescriptor {
        reserved: 0,
        size: std::mem::size_of::<Block>() as u64,
    };

    // BLOCK_IS_GLOBAL (1<<28) — a stateless, statically-allocated block.
    const BLOCK_IS_GLOBAL: c_int = 1 << 28;

    // Symbol handles resolved once; only the residency accessors are needed inside the block.
    struct StateAccessors {
        get_group: IOReportChannelGetGroupFn,
        get_channel: IOReportChannelGetChannelNameFn,
        state_count: IOReportStateGetCountFn,
        state_name: IOReportStateGetNameForIndexFn,
        state_resid: IOReportStateGetResidencyFn,
        get_cstring: CFStringGetCStringFn,
        cf_release: CFReleaseFn,
    }
    static mut ACCESSORS: Option<StateAccessors> = None;

    fn cfstr_to_string(get_cstring: CFStringGetCStringFn, s: CFStringRef) -> String {
        if s.is_null() {
            return String::new();
        }
        let mut buf = [0i8; 128];
        let ok = unsafe { get_cstring(s, buf.as_mut_ptr(), buf.len() as c_longlong, CF_UTF8) };
        if !ok {
            return String::new();
        }
        let cstr = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) };
        cstr.to_string_lossy().into_owned()
    }

    // The iterate callback: for the "GPU Stats" / "GPU Performance States" channel,
    // accumulate (idle_residency, total_residency) deltas into the thread-local.
    extern "C" fn iterate_block(_blk: *mut Block, ch: IOReportSampleRef) -> c_int {
        let acc = unsafe {
            let ptr = std::ptr::addr_of!(ACCESSORS);
            match &*ptr {
                Some(a) => a,
                None => return K_IO_REPORT_ITER_OK,
            }
        };
        let group = cfstr_to_string(acc.get_cstring, unsafe { (acc.get_group)(ch) });
        let channel = cfstr_to_string(acc.get_cstring, unsafe { (acc.get_channel)(ch) });
        let n = unsafe { (acc.state_count)(ch) };
        // The canonical GPU active-residency channel is "GPUPH" (group "GPU Stats",
        // subgroup "GPU Performance States"): a 16-state P-state residency channel whose
        // state[0] is "OFF"/idle and P1..Pn are active clock levels. This is the exact
        // channel `macmon`/`powermetrics` read for GPU utilisation.
        if group != "GPU Stats" || channel != "GPUPH" || n <= 0 {
            return K_IO_REPORT_ITER_OK;
        }
        let (mut idle, mut total) = (0i64, 0i64);
        for i in 0..n {
            let name_ref = unsafe { (acc.state_name)(ch, i) };
            let name = cfstr_to_string(acc.get_cstring, name_ref);
            unsafe { (acc.cf_release)(name_ref) };
            let resid = unsafe { (acc.state_resid)(ch, i) };
            total += resid;
            // Idle / off states are named "IDLE" / "OFF" / "DOWN" on Apple silicon.
            let up = name.to_ascii_uppercase();
            if up.contains("IDLE") || up.contains("OFF") || up.contains("DOWN") {
                idle += resid;
            }
        }
        ACC.with(|c| {
            let (pi, pt) = c.get();
            c.set((pi + idle, pt + total));
        });
        K_IO_REPORT_ITER_OK
    }

    static ITER_BLOCK: Block = Block {
        isa: unsafe { _NSConcreteGlobalBlock.as_ptr() as *const c_void },
        flags: BLOCK_IS_GLOBAL,
        reserved: 0,
        invoke: iterate_block,
        descriptor: &BLOCK_DESCRIPTOR,
    };

    unsafe fn sym<T>(handle: *mut c_void, name: &[u8]) -> Option<T> {
        let p = dlsym(handle, name.as_ptr() as *const c_char);
        if p.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy::<*mut c_void, T>(&p))
        }
    }

    impl GpuUtil {
        /// Resolve IOReport + CF, subscribe to the "GPU Stats" group, and prime the
        /// baseline sample. Returns `None` (util disabled) on any failure.
        pub fn new() -> Option<GpuUtil> {
            unsafe {
                let cf = dlopen(
                    b"/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation\0"
                        .as_ptr() as *const c_char,
                    RTLD_NOW,
                );
                // IOReport's symbols live in /usr/lib/libIOReport.dylib (the framework
                // bundle path is not dlopen-able — it is cache-only under a different
                // install name). Fall back to the framework path just in case.
                let mut ior = dlopen(
                    b"/usr/lib/libIOReport.dylib\0".as_ptr() as *const c_char,
                    RTLD_NOW,
                );
                if ior.is_null() {
                    ior = dlopen(
                        b"/System/Library/PrivateFrameworks/IOReport.framework/IOReport\0"
                            .as_ptr() as *const c_char,
                        RTLD_NOW,
                    );
                }
                if cf.is_null() || ior.is_null() {
                    return None;
                }

                let cfstr_create: CFStringCreateWithCStringFn =
                    sym(cf, b"CFStringCreateWithCString\0")?;
                let cf_get_cstring: CFStringGetCStringFn = sym(cf, b"CFStringGetCString\0")?;
                let cf_release: CFReleaseFn = sym(cf, b"CFRelease\0")?;

                let copy_channels: IOReportCopyChannelsInGroupFn =
                    sym(ior, b"IOReportCopyChannelsInGroup\0")?;
                let create_sub: IOReportCreateSubscriptionFn =
                    sym(ior, b"IOReportCreateSubscription\0")?;
                let create_samples: IOReportCreateSamplesFn =
                    sym(ior, b"IOReportCreateSamples\0")?;
                let create_delta: IOReportCreateSamplesDeltaFn =
                    sym(ior, b"IOReportCreateSamplesDelta\0")?;
                let iterate: IOReportIterateFn = sym(ior, b"IOReportIterate\0")?;
                let get_group: IOReportChannelGetGroupFn =
                    sym(ior, b"IOReportChannelGetGroup\0")?;
                let get_channel: IOReportChannelGetChannelNameFn =
                    sym(ior, b"IOReportChannelGetChannelName\0")?;
                let state_count: IOReportStateGetCountFn =
                    sym(ior, b"IOReportStateGetCount\0")?;
                let state_name: IOReportStateGetNameForIndexFn =
                    sym(ior, b"IOReportStateGetNameForIndex\0")?;
                let state_resid: IOReportStateGetResidencyFn =
                    sym(ior, b"IOReportStateGetResidency\0")?;

                let group = cfstr_create(
                    std::ptr::null(),
                    b"GPU Stats\0".as_ptr() as *const c_char,
                    CF_UTF8,
                );
                if group.is_null() {
                    return None;
                }
                let channels = copy_channels(group, std::ptr::null(), 0, 0, 0);
                cf_release(group);
                if channels.is_null() {
                    return None;
                }

                let mut subbed: CFMutableDictionaryRef = std::ptr::null_mut();
                let sub = create_sub(
                    std::ptr::null_mut(),
                    channels,
                    &mut subbed as *mut CFMutableDictionaryRef,
                    0,
                    std::ptr::null(),
                );
                if sub.is_null() {
                    cf_release(channels);
                    return None;
                }

                // Publish the residency accessors for the iterate block, then prime.
                let ptr = std::ptr::addr_of_mut!(ACCESSORS);
                *ptr = Some(StateAccessors {
                    get_group,
                    get_channel,
                    state_count,
                    state_name,
                    state_resid,
                    get_cstring: cf_get_cstring,
                    cf_release,
                });

                let prev = create_samples(sub, channels, std::ptr::null());
                if prev.is_null() {
                    cf_release(channels);
                    return None;
                }

                Some(GpuUtil {
                    sub,
                    channels,
                    prev,
                    cf_release,
                    create_samples,
                    create_delta,
                    iterate,
                })
            }
        }

        /// Take a fresh sample, delta it against the previous, and iterate the delta to
        /// compute GPU active-residency %. Returns `None` if the delta had no GPU state
        /// residency (e.g. no work happened between samples).
        pub fn sample(&mut self) -> Option<Reading> {
            unsafe {
                let cur = (self.create_samples)(self.sub, self.channels, std::ptr::null());
                if cur.is_null() {
                    return None;
                }
                let delta = (self.create_delta)(self.prev, cur, std::ptr::null());
                (self.cf_release)(self.prev);
                self.prev = cur;
                if delta.is_null() {
                    return None;
                }

                ACC.with(|c| c.set((0, 0)));
                (self.iterate)(delta, &ITER_BLOCK as *const Block as *const c_void);
                (self.cf_release)(delta);

                let (idle, total) = ACC.with(|c| c.get());
                if total <= 0 {
                    return None;
                }
                let active = (total - idle).max(0) as f64;
                Some(Reading {
                    active_pct: (active / total as f64) * 100.0,
                    freq_mhz: None,
                })
            }
        }
    }

    impl Drop for GpuUtil {
        fn drop(&mut self) {
            unsafe {
                if !self.prev.is_null() {
                    (self.cf_release)(self.prev);
                }
                if !self.channels.is_null() {
                    (self.cf_release)(self.channels as CFTypeRef);
                }
                if !self.sub.is_null() {
                    (self.cf_release)(self.sub);
                }
            }
        }
    }
}
