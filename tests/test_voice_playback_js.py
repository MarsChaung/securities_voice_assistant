import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_voice_playback_scheduler() -> None:
    test_file = Path(__file__).parent / "js" / "test_voice_playback.js"

    subprocess.run(
        ["node", "--test", str(test_file)],
        check=True,
        capture_output=True,
        text=True,
    )
