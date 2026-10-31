"""Launch vLLM-Omni with model inspection in the server process.

The pinned Ascend image aborts while the short-lived vLLM registry helper is
tearing torch-npu down.  Model inspection itself succeeds.  Running the same
inspection function in the long-lived API server avoids that image bug without
editing either shared checkout or changing the serving execution path.
"""

from __future__ import annotations

import runpy

from vllm.model_executor.models import registry


def main() -> None:
    registry._run_in_subprocess = lambda function: function()
    runpy.run_module("vllm_omni.entrypoints.cli.main", run_name="__main__")


if __name__ == "__main__":
    main()
