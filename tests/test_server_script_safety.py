"""Static regression guards for destructive live-server test cleanup.

The scripts require a real ComfyUI/GPU stack and are not unit-runnable, but
their process-ownership rule is still testable: a PID discovered from port
8188 must never replace the PID returned by this script's own Popen call.
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPTS = [
    "test_server_memory_trend_phase17.py",
    "test_ref2video_memory_trend.py",
    "test_ref2video_server_e2e.py",
    "test_ref2video_server_hit.py",
]


@pytest.mark.parametrize("script_name", SERVER_SCRIPTS)
def test_live_server_script_never_adopts_an_existing_listener_pid(script_name):
    source = (REPO_ROOT / "scripts" / script_name).read_text()

    assert "= next(iter(bound" not in source
    assert "using the bound PID" not in source
    assert "refusing to adopt or stop it" in source

