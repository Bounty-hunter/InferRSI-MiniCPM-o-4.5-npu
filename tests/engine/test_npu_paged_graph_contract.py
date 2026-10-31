"""Source contracts for the optional Ascend paged-attention graph path."""

import ast
from pathlib import Path


def _method_source(class_name: str, method_name: str) -> str:
    path = Path(__file__).parents[2] / "engine" / "runtime" / "npu_paged.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"missing {class_name}.{method_name}")


def test_paged_graph_replay_reuses_capture_workspace():
    source = _method_source("NpuDecodeGraph", "_update_attention")
    capture = _method_source("NpuDecodeGraph", "_capture_exclusive")
    init_source = _method_source("NpuDecodeGraph", "__init__")
    attention_source = _method_source("_PagedGraphCapture", "attention")

    assert "_npu_paged_attention_get_workspace" not in source
    assert "workspace=self.workspace" in source
    assert "_npu_paged_attention_get_workspace" not in capture
    assert "_npu_paged_attention_get_workspace" in init_source
    assert "workspace=self.workspace" in attention_source
    assert "_npu_paged_attention_get_workspace" not in attention_source
    workspace_context_line = next(
        line for line in init_source.splitlines() if "self.workspace_context_lens = torch.full" in line
    )
    assert "device=" not in workspace_context_line
    assert "_PA_MAX_WORKSPACE_SEQ_LEN = 6144" in (
        Path(__file__).parents[2] / "engine" / "runtime" / "npu_paged.py"
    ).read_text(encoding="utf-8")


def test_graph_ring_retries_only_transient_first_capture_failures():
    source = _method_source("NpuDecoderRunner", "forward_graph_lane")

    assert "_GRAPH_CAPTURE_ATTEMPTS" in source
    assert "_is_transient_graph_update_error" in source
    assert "graph_ring_capture_retries" in source
    assert "captured_output = None" in source
    assert "if captured_output is not None" in source
    assert "ring[graph_lane].run" in source


def test_transient_graph_update_classifier_is_narrow():
    path = Path(__file__).parents[2] / "engine" / "runtime" / "npu_paged.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_is_transient_graph_update_error"
    )
    function_source = ast.get_source_segment(source, function) or ""

    assert '"graph_task_update_end" in message' in function_source
    assert '"507000" in message' in function_source
    assert " and " in function_source


def test_static_fia_graph_uses_device_mask_without_task_updates():
    prepare = _method_source("_StaticFiaGraphCapture", "prepare")
    attention = _method_source("_StaticFiaGraphCapture", "attention")
    kwargs = _method_source("NpuDecodeGraph", "_static_attention_kwargs")
    run = _method_source("NpuDecodeGraph", "run")

    assert "seq_lens_device" in prepare
    assert "torch.where" in prepare
    assert "pse_shift" in kwargs
    assert '"actual_seq_lengths_kv": [self.static_capacity] * self.batch' in kwargs
    assert "npu_fused_infer_attention_score.out" in attention
    assert "graph_task_update" not in prepare + attention + kwargs
    assert "if self.static_capacity is None" in run


def test_static_fia_keeps_task_update_fallback():
    forward = _method_source("NpuDecoderRunner", "forward")
    lane = _method_source("NpuDecoderRunner", "forward_graph_lane")
    source = (
        Path(__file__).parents[2] / "engine" / "runtime" / "npu_paged.py"
    ).read_text(encoding="utf-8")

    assert "static_shape_attention: bool = False" in source
    assert "_static_capacity_for" in forward
    assert "NpuDecodeGraph(self, bucket, self.graph_max_blocks)" in forward
    assert "static_capacity=static_capacity" in lane
    assert "return self.forward(x, layout)" in lane


def test_static_fia_reuses_one_graph_for_every_k_lane():
    lane = _method_source("NpuDecoderRunner", "forward_graph_lane")

    assert "graph_lanes = 1 if static_capacity is not None else int(lanes)" in lane
    assert "graph_lane = 0 if static_capacity is not None else int(lane)" in lane
    assert "ring[graph_lane].run" in lane
