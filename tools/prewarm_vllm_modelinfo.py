"""Populate vLLM's model-info cache without its short-lived subprocess.

The current Ascend image aborts in glibc teardown after the registry helper
has imported torch-npu.  Inspection itself succeeds; performing it in-process
allows vLLM to persist the content-addressed cache before that teardown bug.
This only writes the runner user's normal vLLM cache and never edits the
shared vLLM/vLLM-Omni checkout.
"""

from __future__ import annotations

import json

from vllm.model_executor.models import registry
from vllm.plugins import load_general_plugins


def main() -> None:
    load_general_plugins()
    registry._run_in_subprocess = lambda function: function()
    model_class, architecture = registry.ModelRegistry.inspect_model_cls(
        ["MiniCPMO45OmniForConditionalGeneration"]
    )
    print(json.dumps({"architecture": architecture, "model_class": f"{model_class.__module__}:{model_class.__name__}"}))


if __name__ == "__main__":
    main()
