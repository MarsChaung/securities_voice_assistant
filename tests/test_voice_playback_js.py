import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize(
    "filename",
    ["test_voice_playback.js", "test_voice_barge_in.js"],
)
def test_browser_voice_modules(filename: str) -> None:
    test_file = Path(__file__).parent / "js" / filename
    subprocess.run(
        ["node", "--test", str(test_file)],
        check=True,
        capture_output=True,
        text=True,
    )
