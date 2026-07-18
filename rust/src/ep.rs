//! `MlxEp` — our `OrtEp` C-ABI vtable, generalized from the single-Add spike into a real engine:
//!
//!   * GetCapability claims nodes via the registry claim predicates and groups them into maximal
//!     convex connected clusters (`build_convex_clusters`, a faithful port of ep.cc's union-find +
//!     reachability-bitset algorithm — non-convex fusion creates a cycle ORT rejects).
//!   * Compile extracts each node's `NodeDesc` (op_type/domain/since_version + attributes + I/O
//!     tensor refs) and builds one `Plan` per fused subgraph, owned by its `OrtNodeComputeInfo`.
//!   * Compute (RunPlan) resolves subgraph inputs from the KernelContext, runs each node's handler
//!     in topo order, one `mlx_eval`, and writes each subgraph output.
//!
//! Raw `unsafe`/FFI is confined to this boundary layer + `sys`; the ops use the safe `Array` wrappers.

use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

use crate::engine::{InitData, NodeDesc, OutRef, Plan, Slot, Src, SubgraphDesc, TensorRef, TranslationContext};
use crate::factory::ORT_API_VERSION;
use crate::mlx::Stream;
use crate::registry::{claimable, NodeView};
use crate::sys::{mlx, ort};

#[repr(C)]
pub struct MlxEp {
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    name: CString,
    stream: Stream,
}

impl MlxEp {
    pub fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        name: &CStr,
        _logger: *const ort::OrtLogger,
    ) -> Box<MlxEp> {
        let mut base: ort::OrtEp = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetDefaultMemoryDevice = Some(get_default_memory_device);
        Box::new(MlxEp {
            base,
            ort_api,
            ep_api,
            name: name.to_owned(),
            stream: Stream::new_default_gpu(),
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self) as *mut ort::OrtEp
    }
}

// The per-EP mlx stream is now owned by the `Stream` RAII wrapper, freed exactly once when ORT
// calls ReleaseEp (which drops our Box<MlxEp>). No manual free / no explicit Drop needed.

// On EP teardown, flush the (env-gated) trace. The tracer's collector accumulates
// across all sessions in the process, so each teardown rewrites the full cumulative
// trace; the last one leaves the complete file on disk (no-op when tracing is off).
impl Drop for MlxEp {
    fn drop(&mut self) {
        let tr = crate::trace::tracer();
        // Compact agent-friendly "slowest ops" summary (stderr + trace metadata) before
        // the JSON is written, so the ranking is embedded in the exported trace too.
        tr.log_slowest_ops();
        // The at-a-glance session digest (claim rate, per-path Compute breakdown, memory movement,
        // time attribution). Printed to stderr when tracing OR the verbose flag is on; also embedded
        // in the JSON trace. No-op / no stderr when neither is set.
        tr.log_summary();
        tr.export();
    }
}

#[inline]
unsafe fn this(p: *const ort::OrtEp) -> *const MlxEp {
    p as *const MlxEp
}

unsafe extern "C" fn get_name(p: *const ort::OrtEp) -> *const c_char {
    unsafe {
        (*this(p)).name.as_ptr()
    }
}

unsafe extern "C" fn get_default_memory_device(
    _p: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> *mut ort::OrtStatus {
    unsafe {
        // I/O stays on the CPU allocator (unified memory); no device memory advertised.
        *device = ptr::null();
        ptr::null_mut()
    }
}

// ---------------------------------------------------------------------------
// GetCapability: claim via registry + convex clustering.
// ---------------------------------------------------------------------------

unsafe extern "C" fn get_capability(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "get_capability", || get_capability_impl(p, graph, support))
    }
}

unsafe fn get_capability_impl(
    p: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    unsafe {
        let ep = &*this(p);
        let api = &*ep.ort_api;
        let ep_api = &*ep.ep_api;

        let mut num: usize = 0;
        let st = (api.Graph_GetNumNodes.unwrap())(graph, &mut num);
        if !st.is_null() {
            return st;
        }
        if num == 0 {
            return ptr::null_mut();
        }
        let mut nodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num];
        let st = (api.Graph_GetNodes.unwrap())(graph, nodes.as_mut_ptr(), num);
        if !st.is_null() {
            return st;
        }

        // Control-flow support: ORT partitions bottom-up, presenting a CF node's body subgraph to
        // GetCapability BEFORE the parent graph that owns the node. If this graph is the body of a
        // control-flow node we can translate WHOLE, decline ALL body nodes here (so ORT leaves the body
        // intact); we then claim the CF node itself at the parent level and translate its body in Compile
        // via Node_GetSubgraphs.
        //
        // If the parent CF op is one we CANNOT translate wholesale, we must ALSO decline every body node:
        // ORT would otherwise fuse the claimed body nodes into a node in the EP's private domain and
        // splice it back into the subgraph, but nested subgraphs carry no opset import for that domain,
        // yielding an INVALID_GRAPH ("No opset import for domain 'MLXExecutionProvider'") at session
        // creation (e.g. the Loop that function-inlined SequenceMap expands to). Such body ops simply
        // run on ORT's CPU control flow instead.
        let mut in_cf_body = false;
        if let Some(get_parent) = api.Graph_GetParentNode {
            let mut parent: *const ort::OrtNode = ptr::null();
            let st = get_parent(graph, &mut parent);
            if st.is_null() && !parent.is_null() {
                in_cf_body = true;
            } else if !st.is_null() {
                release_status(api, st);
            }
        }

        // Which nodes can MLX translate exactly (registry claim predicate).
        let supported: Vec<bool> = nodes
            .iter()
            .map(|&node| {
                if in_cf_body {
                    return false;
                }
                let view = NodeView::new(ep.ort_api, node);
                claimable(&view)
            })
            .collect();

        // Claiming view: build the per-op fallback reasons for the declined nodes (only when
        // observability is active, so this extra FFI never touches the traced-off fast path). The
        // legacy `MLX_EP_CLAIM_DEBUG` env still forces the raw stderr dump for quick debugging.
        let tr = crate::trace::tracer();
        let mut rejected: Vec<(String, usize, String, Vec<String>)> = Vec::new();
        if tr.active() || std::env::var_os("MLX_EP_CLAIM_DEBUG").is_some() {
            use std::collections::BTreeMap;
            // Per op-type: (count, first-reason, up to a few node names for locating them).
            let mut acc: BTreeMap<String, (usize, String, Vec<String>)> = BTreeMap::new();
            for (&node, &ok) in nodes.iter().zip(supported.iter()) {
                if !ok {
                    let view = NodeView::new(ep.ort_api, node);
                    let e = acc.entry(view.op_type()).or_insert((0, String::new(), Vec::new()));
                    e.0 += 1;
                    if e.1.is_empty() {
                        e.1 = if in_cf_body {
                            "inside a control-flow subgraph body — claimed as part of the parent \
                             If/Loop/Scan, not individually"
                                .to_string()
                        } else {
                            crate::registry::claim_decision(&view)
                                .err()
                                .map(|c| c.into_owned())
                                .unwrap_or_else(|| "declined (no reason reported)".to_string())
                        };
                    }
                    if e.2.len() < 16 {
                        let nm = view.name();
                        if !nm.is_empty() {
                            e.2.push(nm);
                        }
                    }
                }
            }
            rejected = acc
                .into_iter()
                .map(|(op, (n, why, names))| (op, n, why, names))
                .collect();
            rejected.sort_by(|a, b| b.1.cmp(&a.1));
            if std::env::var_os("MLX_EP_CLAIM_DEBUG").is_some() {
                for (op, n, why, names) in &rejected {
                    eprintln!("[rust-mlx-ep] unclaimed {op} x{n} ({why}): {names:?}");
                }
            }
        }

        let clusters = build_convex_clusters(api, &nodes, &supported);

        let add_fuse = ep_api.EpGraphSupportInfo_AddNodesToFuse.unwrap();
        let mut claimed = 0usize;
        for cluster in &clusters {
            let group: Vec<*const ort::OrtNode> = cluster.iter().map(|&i| nodes[i]).collect();
            let mut opts: ort::OrtNodeFusionOptions = std::mem::zeroed();
            opts.ort_version_supported = ORT_API_VERSION;
            // ORT supplies constant initializers as runtime fused-node inputs (we read them at Run).
            opts.drop_constant_initializers = false;
            let st = add_fuse(support, group.as_ptr(), group.len(), &opts);
            if !st.is_null() {
                return st;
            }
            claimed += cluster.len();
        }
        // Claiming view: claimed/total nodes, fused-subgraph count (fragmentation signal), and the
        // per-op fallback reasons — structured spans/counters + the session summary (near-zero cost
        // and no stderr spam when tracing is off). Replaces the old unconditional eprintlns.
        tr.record_claim(claimed, num, clusters.len(), &rejected);
        ptr::null_mut()
    }
}

/// Release a non-null `OrtStatus` returned on an error / not-found path (the OrtApi allocates a
/// status object the caller owns even for benign "not found" / "buffer too small" results).
#[inline]
unsafe fn release_status(api: &ort::OrtApi, st: *mut ort::OrtStatus) {
    unsafe {
        if !st.is_null() {
            (api.ReleaseStatus.unwrap())(st);
        }
    }
}

/// Value-info tensor name, or "" for an omitted optional slot.
unsafe fn value_info_name(api: &ort::OrtApi, vi: *const ort::OrtValueInfo) -> String {
    unsafe {
        if vi.is_null() {
            return String::new();
        }
        let mut p: *const c_char = ptr::null();
        let st = (api.GetValueInfoName.unwrap())(vi, &mut p);
        if !st.is_null() {
            release_status(api, st);
            return String::new();
        }
        if p.is_null() {
            return String::new();
        }
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn node_input_names(api: &ort::OrtApi, node: *const ort::OrtNode) -> Vec<String> {
    unsafe {
        let mut n: usize = 0;
        (api.Node_GetNumInputs.unwrap())(node, &mut n);
        let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
        if n > 0 {
            (api.Node_GetInputs.unwrap())(node, v.as_mut_ptr(), n);
        }
        v.iter().map(|&vi| value_info_name(api, vi)).collect()
    }
}

unsafe fn node_output_names(api: &ort::OrtApi, node: *const ort::OrtNode) -> Vec<String> {
    unsafe {
        let mut n: usize = 0;
        (api.Node_GetNumOutputs.unwrap())(node, &mut n);
        let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
        if n > 0 {
            (api.Node_GetOutputs.unwrap())(node, v.as_mut_ptr(), n);
        }
        v.iter().map(|&vi| value_info_name(api, vi)).collect()
    }
}

/// Groups supported nodes into maximal, convex, connected clusters. A set S is convex (a valid single
/// fused node) iff no node x outside S lies on a path between two members of S. Faithful port of
/// `BuildConvexClusters` (union-find + reachability bitsets).
fn build_convex_clusters(
    api: &ort::OrtApi,
    nodes: &[*const ort::OrtNode],
    supported: &[bool],
) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let words = (n + 63) / 64;

    // tensor name -> producing node index.
    let mut producer: HashMap<String, usize> = HashMap::new();
    for (i, &node) in nodes.iter().enumerate() {
        for name in unsafe { node_output_names(api, node) } {
            if !name.is_empty() {
                producer.entry(name).or_insert(i);
            }
        }
    }

    // Direct successors / predecessors within the graph.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let mut seen: HashSet<usize> = HashSet::new();
        for name in unsafe { node_input_names(api, nodes[j]) } {
            if name.is_empty() {
                continue;
            }
            if let Some(&i) = producer.get(&name) {
                if i != j && seen.insert(i) {
                    succ[i].push(j);
                    pred[j].push(i);
                }
            }
        }
    }

    // Kahn topological order for reachability accumulation.
    let mut indeg: Vec<usize> = pred.iter().map(|p| p.len()).collect();
    let mut stack: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(u) = stack.pop() {
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                stack.push(v);
            }
        }
    }
    if order.len() != n {
        order = (0..n).collect();
    }

    // reach[i] = set of nodes reachable from i (transitive successors, excluding i).
    let mut reach: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for &u in order.iter().rev() {
        for &v in &succ[u] {
            bit_set(&mut reach[u], v);
            let src = reach[v].clone();
            bit_or_into(&mut reach[u], &src);
        }
    }

    // Cluster state keyed by union-find root.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut cluster_bits: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    let mut reach_bits: Vec<Vec<u64>> = vec![vec![0u64; words]; n];
    for i in 0..n {
        if supported[i] {
            bit_set(&mut cluster_bits[i], i);
            reach_bits[i] = reach[i].clone();
        }
    }

    // Candidate merge edges: direct data edges between two supported nodes.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for u in 0..n {
        if !supported[u] {
            continue;
        }
        for &v in &succ[u] {
            if supported[v] {
                edges.push((u, v));
            }
        }
    }

    let is_convex = |s_bits: &[u64], reach_s: &[u64], reach: &[Vec<u64>]| -> bool {
        for x in 0..n {
            if bit_test(s_bits, x) {
                continue;
            }
            if !bit_test(reach_s, x) {
                continue; // S cannot reach x
            }
            if bit_intersects(&reach[x], s_bits) {
                return false; // x can reach back into S
            }
        }
        true
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &(a, b) in &edges {
            let ra = uf_find(&mut parent, a);
            let rb = uf_find(&mut parent, b);
            if ra == rb {
                continue;
            }
            let mut merged = cluster_bits[ra].clone();
            bit_or_into(&mut merged, &cluster_bits[rb]);
            let mut merged_reach = reach_bits[ra].clone();
            bit_or_into(&mut merged_reach, &reach_bits[rb]);
            if !is_convex(&merged, &merged_reach, &reach) {
                continue;
            }
            parent[rb] = ra;
            cluster_bits[ra] = merged;
            reach_bits[ra] = merged_reach;
            changed = true;
        }
    }

    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        if supported[i] {
            let root = uf_find(&mut parent, i);
            grouped.entry(root).or_default().push(i);
        }
    }
    let mut clusters: Vec<Vec<usize>> = grouped
        .into_values()
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();

    // ---- quotient-acyclicity guard --------------------------------------------------------------
    // Per-cluster convexity is necessary but NOT sufficient: two individually-convex clusters can
    // still cycle THROUGH the CPU nodes between them (a1->b1 in one direction, b2->a2 in the other),
    // and ORT rejects a cyclic contraction ("the graph is not acyclic"). Contract the clusters into a
    // quotient graph and, while it has a cycle, drop the smallest offending cluster (its nodes fall
    // back to CPU) until acyclic. Terminates (each pass removes one cluster) and is a no-op whenever
    // the partition is already acyclic (the common case).
    loop {
        let mut qid = vec![usize::MAX; n];
        for (ci, cl) in clusters.iter().enumerate() {
            for &node in cl {
                qid[node] = ci;
            }
        }
        let mut next = clusters.len();
        for q in qid.iter_mut() {
            if *q == usize::MAX {
                *q = next;
                next += 1;
            }
        }
        let mut qsucc: Vec<HashSet<usize>> = vec![HashSet::new(); next];
        let mut qindeg = vec![0usize; next];
        for u in 0..n {
            for &v in &succ[u] {
                let (a, b) = (qid[u], qid[v]);
                if a != b && qsucc[a].insert(b) {
                    qindeg[b] += 1;
                }
            }
        }
        let mut stack: Vec<usize> = (0..next).filter(|&i| qindeg[i] == 0).collect();
        let mut visited = 0usize;
        while let Some(u) = stack.pop() {
            visited += 1;
            for &v in &qsucc[u] {
                qindeg[v] -= 1;
                if qindeg[v] == 0 {
                    stack.push(v);
                }
            }
        }
        if visited == next {
            break; // quotient is acyclic
        }
        // A cycle remains: nodes with residual in-degree are on/after it. Drop the smallest CLUSTER
        // among them (super-node id < clusters.len()); its members return to CPU next pass.
        let victim = (0..next)
            .filter(|&i| qindeg[i] > 0 && i < clusters.len())
            .min_by_key(|&ci| clusters[ci].len());
        match victim {
            Some(ci) => {
                clusters.remove(ci);
            }
            None => break, // unreachable: singletons alone cannot cycle in an acyclic base graph
        }
    }

    clusters.sort_by_key(|c| c[0]);
    clusters
}

fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[inline]
fn bit_set(b: &mut [u64], i: usize) {
    b[i >> 6] |= 1u64 << (i & 63);
}
#[inline]
fn bit_test(b: &[u64], i: usize) -> bool {
    (b[i >> 6] >> (i & 63)) & 1 != 0
}
#[inline]
fn bit_or_into(dst: &mut [u64], src: &[u64]) {
    for i in 0..dst.len() {
        dst[i] |= src[i];
    }
}
#[inline]
fn bit_intersects(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| x & y != 0)
}

// ---------------------------------------------------------------------------
// Compile: build one Plan (topo-ordered NodeDescs) per fused subgraph.
// ---------------------------------------------------------------------------

unsafe extern "C" fn compile(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "compile", || {
            compile_impl(p, graphs, fused_nodes, count, node_compute_infos, ep_context_nodes)
        })
    }
}

unsafe fn compile_impl(
    p: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    unsafe {
        let ep = &*this(p);
        let api = &*ep.ort_api;

        for i in 0..count {
            let graph = *graphs.add(i);
            let fused_node = *fused_nodes.add(i);
            match build_plan(api, graph, fused_node) {
                Ok(plan) => {
                    let info = SubgraphComputeInfo::new(ep.ort_api, ep.stream.as_raw(), plan);
                    *node_compute_infos.add(i) = Box::into_raw(info) as *mut ort::OrtNodeComputeInfo;
                }
                Err(msg) => {
                    let c =
                        CString::new(msg).unwrap_or_else(|_| CString::new("MLX compile error").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }
        ptr::null_mut()
    }
}

unsafe fn build_plan(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    fused_node: *const ort::OrtNode,
) -> Result<Plan, String> {
    unsafe {
        // Fused-node input/output name -> OrtKernelContext index (the runtime I/O boundary).
        let ctx_input_index: HashMap<String, usize> = node_input_names(api, fused_node)
            .into_iter()
            .enumerate()
            .filter(|(_, n)| !n.is_empty())
            .map(|(k, n)| (n, k))
            .collect();
        let ctx_output_index: HashMap<String, usize> = node_output_names(api, fused_node)
            .into_iter()
            .enumerate()
            .filter(|(_, n)| !n.is_empty())
            .map(|(k, n)| (n, k))
            .collect();

        // Constant initializers referenced by the subgraph (session-owned storage).
        let initializers = collect_initializers(api, graph)?;

        // Subgraph nodes.
        let mut num_nodes: usize = 0;
        (api.Graph_GetNumNodes.unwrap())(graph, &mut num_nodes);
        let mut snodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
        if num_nodes > 0 {
            (api.Graph_GetNodes.unwrap())(graph, snodes.as_mut_ptr(), num_nodes);
        }

        // Producer of each intra-subgraph tensor.
        let mut producer: HashMap<String, usize> = HashMap::new();
        for (k, &node) in snodes.iter().enumerate() {
            for name in node_output_names(api, node) {
                if !name.is_empty() {
                    producer.entry(name).or_insert(k);
                }
            }
        }

        // Topological order over the subgraph.
        let order = topo_order(api, &snodes, &producer);

        let mut nodes: Vec<NodeDesc> = Vec::with_capacity(snodes.len());
        for &idx in &order {
            let node = snodes[idx];
            let op_type = node_op_type(api, node);
            let domain = node_domain(api, node);
            let since_version = node_since_version(api, node);
            let mut nd = NodeDesc::new(op_type, domain, since_version);

            collect_attributes(api, node, &mut nd);

            // Build-time span so each subgraph's op structure is visible in the trace.
            let _op_span = crate::trace::tracer().op_span(
                &nd.op_type,
                node_input_names(api, node).len(),
                node_output_names(api, node).len(),
            );

            // Inputs.
            for name in node_input_names(api, node) {
                let tr = if name.is_empty() {
                    TensorRef::absent()
                } else if producer.contains_key(&name) {
                    TensorRef {
                        name,
                        source: Src::Intermediate,
                        ctx_index: 0,
                        constant: false,
                        shape_const: false,
                        init: None,
                    }
                } else if let Some(&ci) = ctx_input_index.get(&name) {
                    // A constant ctx input's compile-time init pointer goes stale after Compile; the
                    // `constant` flag lets Resolve wrap/cache it once from live ctx data on first Run.
                    let constant = initializers.contains_key(&name);
                    TensorRef {
                        name,
                        source: Src::CtxInput,
                        ctx_index: ci,
                        constant,
                        shape_const: false,
                        init: None,
                    }
                } else if let Some(init) = initializers.get(&name) {
                    TensorRef {
                        name,
                        source: Src::Initializer,
                        ctx_index: 0,
                        constant: false,
                        shape_const: false,
                        init: Some(init.clone()),
                    }
                } else {
                    return Err(format!("MLX could not resolve subgraph input {name}"));
                };
                nd.inputs.push(tr);
            }

            // Outputs.
            for name in node_output_names(api, node) {
                let otype = output_element_type(api, node, &name);
                let (external, ctx_index) = match ctx_output_index.get(&name) {
                    Some(&ci) if !name.is_empty() => (true, ci),
                    _ => (false, 0),
                };
                nd.outputs.push(OutRef {
                    name,
                    external,
                    ctx_index,
                    otype,
                });
            }

            // Control-flow node (If/Scan/Loop): recursively capture its body subgraphs so the handler
            // can translate them inline. Implicit body inputs bottom out at this fused node's ctx
            // boundary (ctx_input_index) or an intra-cluster producer (an enclosing runtime
            // intermediate); body initializers layer over the fused graph's initializers.
            let mut has_subgraphs: usize = 0;
            (api.Node_GetNumSubgraphs.unwrap())(node, &mut has_subgraphs);
            if has_subgraphs > 0 {
                let enclosing_names: HashSet<String> = producer.keys().cloned().collect();
                nd.subgraphs =
                    build_subgraphs(api, node, &ctx_input_index, &enclosing_names, &initializers)?;
            }

            nodes.push(nd);
        }

        // ---- shape-const taint ------------------------------------------------------------------
        // A tensor is "shape-const" if its VALUE is a pure function of input SHAPES + constants (no
        // runtime DATA): a `Shape`/`Size` output (const even when its input is a tracer — it reads
        // only the fixed-per-shape-key shape), an initializer, or any deterministic op whose inputs
        // are all shape-const. Such a value is a real constant inside the mlx_compile trace (no tracer
        // dependency), so a reshape/expand/slice/range target built from it can be eval'd mid-trace
        // and used as a static shape. `nodes` are topologically ordered (producers precede consumers).
        {
            let mut sc: std::collections::HashSet<String> = initializers.keys().cloned().collect();
            let is_random = |op: &str| {
                matches!(
                    op,
                    "RandomNormal" | "RandomUniform" | "RandomNormalLike" | "RandomUniformLike"
                        | "Bernoulli" | "Multinomial" | "Dropout"
                )
            };
            for nd in nodes.iter_mut() {
                for tr in nd.inputs.iter_mut() {
                    if !tr.name.is_empty() && sc.contains(&tr.name) {
                        tr.shape_const = true;
                    }
                }
                let input_const = |tr: &TensorRef| {
                    matches!(tr.source, Src::Absent | Src::Initializer)
                        || tr.constant
                        || tr.shape_const
                };
                let out_const = nd.subgraphs.is_empty()
                    && (matches!(nd.op_type.as_str(), "Shape" | "Size")
                        || (!is_random(&nd.op_type) && nd.inputs.iter().all(input_const)));
                if out_const {
                    for o in &nd.outputs {
                        if !o.name.is_empty() {
                            sc.insert(o.name.clone());
                        }
                    }
                }
            }
        }

        // Compiled-decode fast path (mlx_compile) is allowed unless a control-flow node is present
        // (its graph structure depends on runtime data) or the kill-switch env is set. Detected
        // recursively over the captured body subgraphs.
        fn any_control_flow(nodes: &[NodeDesc]) -> bool {
            nodes.iter().any(|n| {
                !n.subgraphs.is_empty()
                    || n.subgraphs.iter().any(|sg| any_control_flow(&sg.nodes))
            })
        }
        let has_control_flow = any_control_flow(&nodes);
        let mut plan = Plan::new(nodes);
        plan.compiled.enabled = crate::compiled::compile_enabled(has_control_flow);
        plan.prefill.enabled =
            crate::compiled::prefill_enabled(has_control_flow, &plan.nodes);
        plan.general.enabled =
            crate::compiled::general_enabled(has_control_flow, &plan.nodes);
        Ok(plan)
    }
}

unsafe fn collect_initializers(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
) -> Result<HashMap<String, InitData>, String> {
    unsafe {
        let mut map = HashMap::new();
        let mut num: usize = 0;
        (api.Graph_GetNumInitializers.unwrap())(graph, &mut num);
        if num == 0 {
            return Ok(map);
        }
        let mut vis: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num];
        (api.Graph_GetInitializers.unwrap())(graph, vis.as_mut_ptr(), num);
        for &vi in &vis {
            let name = value_info_name(api, vi);
            if name.is_empty() {
                continue;
            }
            let mut value: *const ort::OrtValue = ptr::null();
            let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
            if !st.is_null() {
                release_status(api, st);
                continue;
            }
            if value.is_null() {
                continue;
            }
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
            (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
            let mut nd: usize = 0;
            (api.GetDimensionsCount.unwrap())(info, &mut nd);
            let mut dims = vec![0i64; nd];
            if nd > 0 {
                (api.GetDimensions.unwrap())(info, dims.as_mut_ptr(), nd);
            }
            let mut etype: ort::ONNXTensorElementDataType = 0;
            (api.GetTensorElementType.unwrap())(info, &mut etype);
            let mut count: usize = 0;
            (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
            (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
            let mut data: *const c_void = ptr::null();
            (api.GetTensorData.unwrap())(value, &mut data);
            map.insert(
                name,
                InitData {
                    data,
                    shape: dims,
                    dtype: etype,
                    count,
                    owned: None,
                },
            );
        }
        Ok(map)
    }
}

/// Ensure an `InitData`'s bytes live in owned (`Arc`-backed) storage. Entries that already own
/// their bytes — or whose data is null / has an unknown element width — are cloned as-is. Enclosing
/// scope initializers captured with `owned: None` point into transient ORT graph storage that does
/// not survive to execute-time control-flow body translation, so their bytes are copied here.
///
/// # Safety
/// `src.data` must point to at least `src.count * element_byte_size(src.dtype)` valid bytes for the
/// duration of this call (true at compile time, when the enclosing initializers were just collected).
unsafe fn own_init_data(src: &InitData) -> InitData {
    if src.owned.is_some() || src.data.is_null() {
        return src.clone();
    }
    let width = element_byte_size(src.dtype);
    if width == 0 {
        return src.clone();
    }
    let nbytes = src.count * width;
    let owned: std::sync::Arc<Vec<u8>> =
        std::sync::Arc::new(unsafe { std::slice::from_raw_parts(src.data as *const u8, nbytes) }.to_vec());
    let data = owned.as_ptr() as *const c_void;
    InitData {
        data,
        shape: src.shape.clone(),
        dtype: src.dtype,
        count: src.count,
        owned: Some(owned),
    }
}

/// Element byte width for an ONNX tensor element type (0 = unsupported). Mirrors ep.cc's
/// `ElementByteSize`, used to copy control-flow body initializer bytes into owned storage.
fn element_byte_size(t: ort::ONNXTensorElementDataType) -> usize {
    match t {
        x if x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 =>
        {
            8
        }
        x if x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 =>
        {
            4
        }
        x if x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 =>
        {
            2
        }
        x if x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
            || x == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL =>
        {
            1
        }
        _ => 0,
    }
}

/// The formal input/output value names of a body graph.
unsafe fn graph_value_names(
    api: &ort::OrtApi,
    graph: *const ort::OrtGraph,
    count_fn: unsafe extern "C" fn(*const ort::OrtGraph, *mut usize) -> *mut ort::OrtStatus,
    get_fn: unsafe extern "C" fn(
        *const ort::OrtGraph,
        *mut *const ort::OrtValueInfo,
        usize,
    ) -> *mut ort::OrtStatus,
) -> Vec<String> {
    unsafe {
        let mut num: usize = 0;
        count_fn(graph, &mut num);
        if num == 0 {
            return Vec::new();
        }
        let mut vis: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num];
        get_fn(graph, vis.as_mut_ptr(), num);
        vis.iter().map(|&vi| value_info_name(api, vi)).collect()
    }
}

/// Recursively build the `SubgraphDesc` list for a control-flow node's body subgraphs (If/Scan/Loop).
/// Faithful port of ep.cc's `BuildSubgraphs`. `ctx_input_index` maps names surfaced at the fused
/// node's runtime I/O boundary; `enclosing_names` are names that resolve as runtime intermediates
/// from an enclosing scope (fused-cluster producers + enclosing formal inputs/producers); a body
/// reference to one is a plain `Src::Intermediate` lookup. `enclosing_inits` are constant
/// initializers visible from enclosing scopes.
unsafe fn build_subgraphs(
    api: &ort::OrtApi,
    cf_node: *const ort::OrtNode,
    ctx_input_index: &HashMap<String, usize>,
    enclosing_names: &HashSet<String>,
    enclosing_inits: &HashMap<String, InitData>,
) -> Result<Vec<SubgraphDesc>, String> {
    unsafe {
        let mut num_subs: usize = 0;
        (api.Node_GetNumSubgraphs.unwrap())(cf_node, &mut num_subs);
        if num_subs == 0 {
            return Ok(Vec::new());
        }
        let mut sub_graphs: Vec<*const ort::OrtGraph> = vec![ptr::null(); num_subs];
        let mut attr_names: Vec<*const c_char> = vec![ptr::null(); num_subs];
        (api.Node_GetSubgraphs.unwrap())(
            cf_node,
            sub_graphs.as_mut_ptr(),
            num_subs,
            attr_names.as_mut_ptr(),
        );

        let mut out: Vec<SubgraphDesc> = Vec::with_capacity(num_subs);
        for si in 0..num_subs {
            let body = sub_graphs[si];
            let attr_name = if attr_names[si].is_null() {
                String::new()
            } else {
                CStr::from_ptr(attr_names[si]).to_string_lossy().into_owned()
            };

            let input_names = graph_value_names(
                api,
                body,
                api.Graph_GetNumInputs.unwrap(),
                api.Graph_GetInputs.unwrap(),
            );
            let output_names = graph_value_names(
                api,
                body,
                api.Graph_GetNumOutputs.unwrap(),
                api.Graph_GetOutputs.unwrap(),
            );

            // Body initializers layered over the enclosing ones (a body may shadow an outer name). Bytes
            // are COPIED into owned storage — the body graph handle is released when this walk returns.
            // The enclosing initializers arrive with `owned: None`: their `data` pointer aims into
            // transient ORT graph storage (Constant-node-folded initializers are re-materialised per
            // query and are NOT stable to execute time). Control-flow bodies are translated lazily at
            // EXECUTE time (the taken If branch, each Scan/Loop step), long after this compile-time walk,
            // so any such pointer would dangle. Copy them into owned storage now so translate-time reads
            // (shape/axes/indices operands like Squeeze `axes`) see the correct bytes at run time.
            let mut inits: HashMap<String, InitData> =
                enclosing_inits.iter().map(|(k, v)| (k.clone(), own_init_data(v))).collect();
            let mut num_init: usize = 0;
            (api.Graph_GetNumInitializers.unwrap())(body, &mut num_init);
            if num_init > 0 {
                let mut vis: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); num_init];
                (api.Graph_GetInitializers.unwrap())(body, vis.as_mut_ptr(), num_init);
                for &vi in &vis {
                    let name = value_info_name(api, vi);
                    if name.is_empty() {
                        continue;
                    }
                    let mut value: *const ort::OrtValue = ptr::null();
                    let st = (api.ValueInfo_GetInitializerValue.unwrap())(vi, &mut value);
                    if !st.is_null() {
                        release_status(api, st);
                        continue;
                    }
                    if value.is_null() {
                        continue;
                    }
                    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
                    (api.GetTensorTypeAndShape.unwrap())(value, &mut info);
                    let mut ndims: usize = 0;
                    (api.GetDimensionsCount.unwrap())(info, &mut ndims);
                    let mut dims = vec![0i64; ndims];
                    if ndims > 0 {
                        (api.GetDimensions.unwrap())(info, dims.as_mut_ptr(), ndims);
                    }
                    let mut etype: ort::ONNXTensorElementDataType = 0;
                    (api.GetTensorElementType.unwrap())(info, &mut etype);
                    let mut count: usize = 0;
                    (api.GetTensorShapeElementCount.unwrap())(info, &mut count);
                    (api.ReleaseTensorTypeAndShapeInfo.unwrap())(info);
                    let width = element_byte_size(etype);
                    let mut raw: *const c_void = ptr::null();
                    (api.GetTensorData.unwrap())(value, &mut raw);
                    if width == 0 || raw.is_null() {
                        continue;
                    }
                    let nbytes = count * width;
                    let owned: std::sync::Arc<Vec<u8>> = std::sync::Arc::new(
                        std::slice::from_raw_parts(raw as *const u8, nbytes).to_vec(),
                    );
                    let data = owned.as_ptr() as *const c_void;
                    inits.insert(
                        name,
                        InitData {
                            data,
                            shape: dims,
                            dtype: etype,
                            count,
                            owned: Some(owned),
                        },
                    );
                }
            }

            // Body nodes + producer/formal sets.
            let mut num_nodes: usize = 0;
            (api.Graph_GetNumNodes.unwrap())(body, &mut num_nodes);
            let mut bnodes: Vec<*const ort::OrtNode> = vec![ptr::null(); num_nodes];
            if num_nodes > 0 {
                (api.Graph_GetNodes.unwrap())(body, bnodes.as_mut_ptr(), num_nodes);
            }
            let mut producer: HashMap<String, usize> = HashMap::new();
            for (k, &bn) in bnodes.iter().enumerate() {
                for name in node_output_names(api, bn) {
                    if !name.is_empty() {
                        producer.entry(name).or_insert(k);
                    }
                }
            }
            let formal: HashSet<String> = input_names.iter().filter(|n| !n.is_empty()).cloned().collect();

            // Names visible to a NESTED control-flow node inside this body: enclosing ∪ formal ∪ producers.
            let mut child_enclosing = enclosing_names.clone();
            child_enclosing.extend(formal.iter().cloned());
            child_enclosing.extend(producer.keys().cloned());

            let order = topo_order(api, &bnodes, &producer);

            let mut nodes: Vec<NodeDesc> = Vec::with_capacity(bnodes.len());
            for &idx in &order {
                let node = bnodes[idx];
                let mut mnd = NodeDesc::new(
                    node_op_type(api, node),
                    node_domain(api, node),
                    node_since_version(api, node),
                );
                collect_attributes(api, node, &mut mnd);

                for name in node_input_names(api, node) {
                    let tr = if name.is_empty() {
                        TensorRef::absent()
                    } else if producer.contains_key(&name) || formal.contains(&name) {
                        TensorRef {
                            name,
                            source: Src::Intermediate,
                            ctx_index: 0,
                            constant: false,
                            shape_const: false,
                            init: None,
                        }
                    } else if let Some(init) = inits.get(&name) {
                        TensorRef {
                            name,
                            source: Src::Initializer,
                            ctx_index: 0,
                            constant: false,
                            shape_const: false,
                            init: Some(init.clone()),
                        }
                    } else if let Some(&ci) = ctx_input_index.get(&name) {
                        TensorRef {
                            name,
                            source: Src::CtxInput,
                            ctx_index: ci,
                            constant: false,
                            shape_const: false,
                            init: None,
                        }
                    } else if enclosing_names.contains(&name) {
                        TensorRef {
                            name,
                            source: Src::Intermediate,
                            ctx_index: 0,
                            constant: false,
                            shape_const: false,
                            init: None,
                        }
                    } else {
                        return Err(format!("MLX could not resolve control-flow body input {name}"));
                    };
                    mnd.inputs.push(tr);
                }

                for name in node_output_names(api, node) {
                    let otype = output_element_type(api, node, &name);
                    // Body outputs are never external ctx outputs.
                    mnd.outputs.push(OutRef {
                        name,
                        external: false,
                        ctx_index: 0,
                        otype,
                    });
                }

                let mut nsub: usize = 0;
                (api.Node_GetNumSubgraphs.unwrap())(node, &mut nsub);
                if nsub > 0 {
                    mnd.subgraphs =
                        build_subgraphs(api, node, ctx_input_index, &child_enclosing, &inits)?;
                }

                nodes.push(mnd);
            }

            out.push(SubgraphDesc {
                attr_name,
                input_names,
                output_names,
                nodes,
            });
        }
        Ok(out)
    }
}

unsafe fn topo_order(
    api: &ort::OrtApi,
    snodes: &[*const ort::OrtNode],
    producer: &HashMap<String, usize>,
) -> Vec<usize> {
    unsafe {
        let n = snodes.len();
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indeg: Vec<usize> = vec![0; n];
        for j in 0..n {
            let mut seen: HashSet<usize> = HashSet::new();
            for name in node_input_names(api, snodes[j]) {
                if name.is_empty() {
                    continue;
                }
                if let Some(&i) = producer.get(&name) {
                    if i != j && seen.insert(i) {
                        succ[i].push(j);
                        indeg[j] += 1;
                    }
                }
            }
        }
        let mut stack: Vec<usize> = (0..n).filter(|&k| indeg[k] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        while let Some(u) = stack.pop() {
            order.push(u);
            for &v in &succ[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    stack.push(v);
                }
            }
        }
        if order.len() != n {
            order = (0..n).collect();
        }
        order
    }
}

unsafe fn node_op_type(api: &ort::OrtApi, node: *const ort::OrtNode) -> String {
    unsafe {
        let mut p: *const c_char = ptr::null();
        (api.Node_GetOperatorType.unwrap())(node, &mut p);
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

unsafe fn node_domain(api: &ort::OrtApi, node: *const ort::OrtNode) -> String {
    unsafe {
        let mut p: *const c_char = ptr::null();
        (api.Node_GetDomain.unwrap())(node, &mut p);
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

unsafe fn node_since_version(api: &ort::OrtApi, node: *const ort::OrtNode) -> i32 {
    unsafe {
        let mut v: i32 = 0;
        (api.Node_GetSinceVersion.unwrap())(node, &mut v);
        v
    }
}

/// Element type of node output named `name` (UNDEFINED if not a tensor).
unsafe fn output_element_type(
    api: &ort::OrtApi,
    node: *const ort::OrtNode,
    name: &str,
) -> ort::ONNXTensorElementDataType {
    unsafe {
        let mut n: usize = 0;
        (api.Node_GetNumOutputs.unwrap())(node, &mut n);
        let mut v: Vec<*const ort::OrtValueInfo> = vec![ptr::null(); n];
        if n > 0 {
            (api.Node_GetOutputs.unwrap())(node, v.as_mut_ptr(), n);
        }
        for &vi in &v {
            if vi.is_null() || value_info_name(api, vi) != name {
                continue;
            }
            let mut ti: *const ort::OrtTypeInfo = ptr::null();
            let st = (api.GetValueInfoTypeInfo.unwrap())(vi, &mut ti);
            if !st.is_null() {
                release_status(api, st);
                return 0;
            }
            if ti.is_null() {
                return 0;
            }
            let mut onnx_type: ort::ONNXType = 0;
            (api.GetOnnxTypeFromTypeInfo.unwrap())(ti, &mut onnx_type);
            if onnx_type != ort::ONNXType_ONNX_TYPE_TENSOR {
                return 0;
            }
            let mut tsi: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
            (api.CastTypeInfoToTensorInfo.unwrap())(ti, &mut tsi);
            if tsi.is_null() {
                return 0;
            }
            let mut dtype: ort::ONNXTensorElementDataType = 0;
            (api.GetTensorElementType.unwrap())(tsi, &mut dtype);
            return dtype;
        }
        0
    }
}

/// Generic attribute copy: every INT/FLOAT/INTS/FLOATS/STRING attr into the NodeDesc maps.
unsafe fn collect_attributes(api: &ort::OrtApi, node: *const ort::OrtNode, nd: &mut NodeDesc) {
    unsafe {
        let mut num: usize = 0;
        (api.Node_GetNumAttributes.unwrap())(node, &mut num);
        if num == 0 {
            return;
        }
        let mut attrs: Vec<*const ort::OrtOpAttr> = vec![ptr::null(); num];
        (api.Node_GetAttributes.unwrap())(node, attrs.as_mut_ptr(), num);
        let read = api.ReadOpAttr.unwrap();
        for &attr in &attrs {
            if attr.is_null() {
                continue;
            }
            let mut name_p: *const c_char = ptr::null();
            (api.OpAttr_GetName.unwrap())(attr, &mut name_p);
            if name_p.is_null() {
                continue;
            }
            let name = CStr::from_ptr(name_p).to_string_lossy().into_owned();
            let mut atype: ort::OrtOpAttrType = 0;
            (api.OpAttr_GetType.unwrap())(attr, &mut atype);
            match atype {
                t if t == ort::OrtOpAttrType_ORT_OP_ATTR_INT => {
                    let mut v: i64 = 0;
                    let mut out: usize = 0;
                    let st = read(
                        attr,
                        atype,
                        &mut v as *mut i64 as *mut c_void,
                        std::mem::size_of::<i64>(),
                        &mut out,
                    );
                    if st.is_null() {
                        nd.ints.insert(name, v);
                    } else {
                        release_status(api, st);
                    }
                }
                t if t == ort::OrtOpAttrType_ORT_OP_ATTR_FLOAT => {
                    let mut v: f32 = 0.0;
                    let mut out: usize = 0;
                    let st = read(
                        attr,
                        atype,
                        &mut v as *mut f32 as *mut c_void,
                        std::mem::size_of::<f32>(),
                        &mut out,
                    );
                    if st.is_null() {
                        nd.floats.insert(name, v);
                    } else {
                        release_status(api, st);
                    }
                }
                t if t == ort::OrtOpAttrType_ORT_OP_ATTR_INTS => {
                    if let Some(v) = read_array::<i64>(api, attr, atype) {
                        nd.int_arrays.insert(name, v);
                    }
                }
                t if t == ort::OrtOpAttrType_ORT_OP_ATTR_FLOATS => {
                    if let Some(v) = read_array::<f32>(api, attr, atype) {
                        nd.float_arrays.insert(name, v);
                    }
                }
                t if t == ort::OrtOpAttrType_ORT_OP_ATTR_STRING => {
                    let mut needed: usize = 0;
                    let probe = read(attr, atype, ptr::null_mut(), 0, &mut needed);
                    release_status(api, probe);
                    if needed > 0 {
                        let mut buf: Vec<u8> = vec![0u8; needed];
                        let mut out: usize = 0;
                        let st = read(attr, atype, buf.as_mut_ptr() as *mut c_void, needed, &mut out);
                        if st.is_null() {
                            buf.truncate(out.min(needed));
                            if let Ok(s) = String::from_utf8(buf) {
                                nd.strings.insert(name, s);
                            }
                        } else {
                            release_status(api, st);
                        }
                    }
                }
                _ => {} // STRINGS / GRAPH / TENSOR not carried by wave-1 ops.
            }
        }
    }
}

/// Read an array-valued attribute (INTS/FLOATS): size, allocate, read.
unsafe fn read_array<T: Copy + Default>(
    api: &ort::OrtApi,
    attr: *const ort::OrtOpAttr,
    atype: ort::OrtOpAttrType,
) -> Option<Vec<T>> {
    unsafe {
        let read = api.ReadOpAttr.unwrap();
        let mut needed_bytes: usize = 0;
        // The size-probe read returns a non-OK status ("result buffer too small") that must be freed.
        let probe = read(attr, atype, ptr::null_mut(), 0, &mut needed_bytes);
        release_status(api, probe);
        if needed_bytes == 0 {
            return Some(Vec::new());
        }
        let elem = std::mem::size_of::<T>();
        let count = needed_bytes / elem;
        let mut buf: Vec<T> = vec![T::default(); count];
        let mut out: usize = 0;
        let st = read(
            attr,
            atype,
            buf.as_mut_ptr() as *mut c_void,
            needed_bytes,
            &mut out,
        );
        if st.is_null() {
            Some(buf)
        } else {
            release_status(api, st);
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fused-subgraph compute info: owns the Plan, runs it through MLX.
// ---------------------------------------------------------------------------

#[repr(C)]
struct SubgraphComputeInfo {
    base: ort::OrtNodeComputeInfo,
    ort_api: *const ort::OrtApi,
    stream: mlx::mlx_stream,
    // ORT permits concurrent Run() on one InferenceSession, and CreateState hands every Run the
    // SAME SubgraphComputeInfo. `plan` is mutated by Compute (the compiled-closure cache is filled
    // on cache-MISS; the eager translator writes intermediates), so it must be serialized to avoid
    // mutable aliasing / a data race. MLX drives a single default stream, so this node is serial
    // regardless — one session per thread remains the path to real cross-request parallelism.
    plan: std::sync::Mutex<Plan>,
    // First thread to run Compute "owns" this session's MLX stream. MLX 0.6.0 eval is thread-affine:
    // a foreign-thread eval aborts the HOST process. We detect a cross-thread Run here and return a
    // clean OrtStatus instead of letting MLX take the process down. Concurrency = session-per-thread.
    owner_thread: std::sync::Mutex<Option<std::thread::ThreadId>>,
}

impl SubgraphComputeInfo {
    fn new(
        ort_api: *const ort::OrtApi,
        stream: mlx::mlx_stream,
        plan: Plan,
    ) -> Box<SubgraphComputeInfo> {
        let mut base: ort::OrtNodeComputeInfo = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.CreateState = Some(create_state);
        base.Compute = Some(compute);
        base.ReleaseState = Some(release_state);
        Box::new(SubgraphComputeInfo {
            base,
            ort_api,
            stream,
            plan: std::sync::Mutex::new(plan),
            owner_thread: std::sync::Mutex::new(None),
        })
    }
}

unsafe extern "C" fn create_state(
    this_ptr: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    unsafe {
        *compute_state = this_ptr as *mut c_void;
        ptr::null_mut()
    }
}

unsafe extern "C" fn release_state(_this: *mut ort::OrtNodeComputeInfo, _state: *mut c_void) {}

unsafe extern "C" fn compute(
    this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*(state as *const SubgraphComputeInfo)).ort_api };
    unsafe { crate::guard_ffi_status(api, "compute", || compute_impl(this, state, kctx)) }
}

unsafe fn compute_impl(
    _this: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
    kctx: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    unsafe {
        let info = &*(state as *const SubgraphComputeInfo);
        let api = &*info.ort_api;

        // Thread-affinity guard: MLX 0.6.0 eval is bound to the thread that first drove this
        // session's stream. A Run from any other thread would abort the host process inside MLX, so
        // bind the owner on first Compute and reject a cross-thread Run with a clean OrtStatus.
        let cur_thread = std::thread::current().id();
        {
            let mut owner = info.owner_thread.lock().unwrap_or_else(|e| e.into_inner());
            match *owner {
                None => *owner = Some(cur_thread),
                Some(t) if t == cur_thread => {}
                Some(t) => {
                    let msg = format!(
                        "onnxruntime-mlx: this InferenceSession first ran on thread {t:?} but Run() \
                         was called from {cur_thread:?}. MLX eval is thread-affine — use one \
                         InferenceSession per thread for concurrent inference."
                    );
                    let c = CString::new(msg).unwrap_or_else(|_| {
                        CString::new("onnxruntime-mlx: cross-thread Run() is not supported").unwrap()
                    });
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // Serialize per-subgraph state: ORT allows concurrent Run() on one session, but the
        // compiled-closure cache (plan.compiled/prefill/general) is mutated on cache-MISS and the
        // eager translator writes intermediates into `plan` — concurrent Compute on the same node
        // must not alias it. `plan_ptr` stays valid for the whole call while the guard is held.
        let mut plan_guard = info.plan.lock().unwrap_or_else(|e| e.into_inner());
        let plan_ptr: *mut Plan = &mut *plan_guard;

        let node_count = (*plan_ptr).nodes.len();
        let tr = crate::trace::tracer();
        tr.note_thread("mlx.ep.compute");
        let _region = tr.subgraph_region(node_count);
        tr.sample_gpu_counters();

        // Compiled-decode fast path: handle single-token (S==1) decode via the once-compiled
        // shapeless closure; prefill (S>1) is handled by the shape-keyed prefill path below, and
        // ineligible plans fall through to the eager translator.
        let seq_len = crate::compiled::detect_seq_len(info.ort_api, kctx, &*plan_ptr);
        if (*plan_ptr).compiled.enabled && seq_len == Some(1) {
            // Cache state: replay (HIT) if the shapeless closure is already compiled, else first
            // trace+compile (MISS). Decode is shapeless, so it never retraces (empty shape key).
            let pre_valid = (*plan_ptr).compiled.valid;
            match crate::compiled::try_compiled(plan_ptr, Slot::Decode, info.ort_api, kctx, info.stream) {
                Ok(true) => {
                    let cache = if pre_valid { crate::trace::CacheState::Hit } else { crate::trace::CacheState::Miss };
                    tr.record_compute_path(crate::trace::ComputePath::Decode, cache, "", node_count);
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled decode failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled decode failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // Compiled-prefill fast path (Phase 2): the SAME decoder subgraph as decode but at query
        // length S>1. `S` bakes into the trace (causal-mask extent, KV write width), so this uses the
        // unified core in SHAPE-KEYED mode — it retraces per distinct prompt length and replays the
        // fused closure for repeats. Declines (=> eager) for any non-decoder / partial-rotary shape.
        if (*plan_ptr).prefill.enabled && matches!(seq_len, Some(s) if s > 1) {
            let pre_valid = (*plan_ptr).prefill.valid;
            match crate::compiled::try_compiled(plan_ptr, Slot::Prefill, info.ort_api, kctx, info.stream) {
                Ok(true) => {
                    // Shape-keyed on the query length S: a changed key means MLX retraced under us.
                    let cache = if pre_valid { crate::trace::CacheState::Hit } else { crate::trace::CacheState::Miss };
                    let key = seq_len.map(|s| format!("S{s}")).unwrap_or_default();
                    tr.record_compute_path(crate::trace::ComputePath::Prefill, cache, &key, node_count);
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled prefill failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled prefill failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        // General compiled fast path: trace + fuse ANY claimed static-shape subgraph (CNN / audio)
        // into a shape-keyed compiled closure and replay it. Declines (=> eager) for attention /
        // control-flow subgraphs and on any trace/apply doubt.
        if (*plan_ptr).general.enabled {
            let pre_valid = (*plan_ptr).general.valid;
            match crate::compiled::try_compiled(
                plan_ptr,
                Slot::General,
                info.ort_api,
                kctx,
                info.stream,
            ) {
                Ok(true) => {
                    let cache = if pre_valid { crate::trace::CacheState::Hit } else { crate::trace::CacheState::Miss };
                    // Shape key from the primary dynamic input so a changed audio/frame size shows as
                    // a RETRACE (only read when observability is active).
                    let key = if tr.active() { compute_shape_key(info.ort_api, kctx) } else { String::new() };
                    tr.record_compute_path(crate::trace::ComputePath::General, cache, &key, node_count);
                    return ptr::null_mut();
                }
                Ok(false) => { /* not eligible — fall back to eager below */ }
                Err(msg) => {
                    let c = CString::new(format!("MLX compiled general failed: {msg}"))
                        .unwrap_or_else(|_| CString::new("MLX compiled general failed").unwrap());
                    return (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr());
                }
            }
        }

        let mut tctx = TranslationContext::new(&mut *plan_ptr, info.ort_api, kctx, info.stream);
        match tctx.execute() {
            Ok(()) => {
                tr.record_compute_path(crate::trace::ComputePath::Eager, crate::trace::CacheState::Na, "", node_count);
                ptr::null_mut()
            }
            Err(msg) => {
                let c = CString::new(format!("MLX subgraph failed: {msg}"))
                    .unwrap_or_else(|_| CString::new("MLX subgraph failed").unwrap());
                (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, c.as_ptr())
            }
        }
    }
}

/// A cheap shape key from the primary (index-0) dynamic ctx input, used to classify a shape-keyed
/// general-subgraph Compute as HIT vs RETRACE. Best-effort: an empty string when the input can't be
/// read. Only called when observability is active.
unsafe fn compute_shape_key(
    ort_api: *const ort::OrtApi,
    kctx: *mut ort::OrtKernelContext,
) -> String {
    match crate::engine::read_ctx_input_raw(ort_api, kctx, 0) {
        Ok((_data, shape, _dtype)) => format!("{shape:?}"),
        Err(_) => String::new(),
    }
}

unsafe extern "C" fn release_node_compute_infos(
    _p: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    num: usize,
) {
    unsafe {
        for i in 0..num {
            let ptr = *infos.add(i);
            if !ptr.is_null() {
                drop(Box::from_raw(ptr as *mut SubgraphComputeInfo));
            }
        }
    }
}
