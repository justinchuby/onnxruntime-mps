"""GatherND (ai.onnx, batch_dims=0) coverage for the MLX EP.

Gathers full slices of ``data`` addressed by the last axis of ``indices``. The output shape is static
(from the operand shapes) so the op is compile-safe; only the index values are runtime. Each case is
proven to run on the MLX EP (per-node profiling) and compared, tolerance-gated, to ORT's CPU EP.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
import pytest
from onnx_ir import DataType as DT

import _models as m


def _indices(ishape, dshape):
    idx = np.zeros(ishape, dtype=np.int64)
    width = ishape[-1]
    for mi in np.ndindex(*ishape[:-1]):
        for j in range(width):
            idx[mi + (j,)] = np.random.randint(-dshape[j], dshape[j])  # exercise negatives
    return idx


def _assert_mlx_claims(model, feeds):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_profiling = True
    opts.profile_file_prefix = "mlx_gnd_probe"
    sess = ort.InferenceSession(model, opts, providers=m.EP_PROVIDERS)
    sess.run(None, feeds)
    path = sess.end_profiling()
    try:
        events = json.load(open(path))
    finally:
        os.remove(path)
    providers = {e.get("args", {}).get("provider") for e in events
                 if e.get("cat") == "Node" and e.get("args", {}).get("provider")}
    assert "MLXExecutionProvider" in providers, f"MLX EP did not claim GatherND (ran on {providers})"


def _check(dshape, ishape, seed=0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    data = rng.standard_normal(dshape).astype(np.float32)
    idx = _indices(ishape, dshape)
    width = ishape[-1]
    oshape = list(ishape[:-1]) + list(dshape[width:])
    model = m.make_model(
        "GatherND",
        [m.tensor("data", DT.FLOAT, list(dshape)), m.tensor("indices", DT.INT64, list(ishape))],
        [m.tensor("out", DT.FLOAT, oshape)],
        attributes={"batch_dims": 0},
    )
    feeds = {"data": data, "indices": idx}
    _assert_mlx_claims(model, feeds)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    expected = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = m.run_mlx(model, feeds)[0]
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dshape,ishape",
    [
        ((2, 2), (2, 2)),        # full index tuples -> rank-1 output
        ((2, 2), (2, 1)),        # slice a row
        ((2, 3, 4), (2, 2)),     # m=2 -> [.., 4]
        ((2, 3, 4), (5, 3)),     # m=3 scalars
        ((4, 5, 6), (2, 3, 1)),  # nested index grid, m=1
        ((3, 4, 5, 6), (2, 2)),  # 3-D tail
        ((8,), (3, 1)),          # 1-D data
    ],
)
def test_gathernd(dshape, ishape):
    _check(dshape, ishape, seed=hash((dshape, ishape)) & 0xFFFF)
