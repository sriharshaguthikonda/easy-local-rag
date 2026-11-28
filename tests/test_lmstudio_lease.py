"""Exercise actual lease ownership with no server or live Foreman state."""

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


BASH = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
LEASE = Path(__file__).resolve().parents[1] / "lmstudio_runtime" / "lease.sh"


@pytest.mark.parametrize("crash", [False, True])
def test_git_bash_lease_preserves_the_reserved_slot(tmp_path, crash):
    node = shutil.which("node")
    if os.name != "nt" or not BASH.is_file() or not node:
        pytest.skip("optional Windows Git Bash and Node runtime")
    bridge = tmp_path / "fake-bridge.mjs"
    bridge.write_text(
        """import { existsSync, writeFileSync } from 'node:fs';
const [mode, marker, permit] = process.argv.slice(2);
if (mode === '--recover') {
  process.exitCode = existsSync(permit) ? 0 : 75;
} else {
  if (process.env.LEASE_TEST_CRASH === '1') writeFileSync(marker, '{}');
  process.stdout.write('ready\\n');
  process.stdin.resume();
  process.stdin.on('end', () => { process.exitCode = 21; });
}
""", encoding="utf-8",
    )
    marker = tmp_path / "slot1" / "pid"
    permit = tmp_path / "permit"
    env = dict(os.environ, MSYS2_ARG_CONV_EXCL="*", LEASE_TEST_CRASH=str(int(crash)))
    args = [str(BASH), "--noprofile", "--norc", LEASE.as_posix(),
            tmp_path.as_posix(), Path(node).as_posix(), bridge.as_posix(), permit.as_posix()]
    owner = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, env=env)
    try:
        until = time.monotonic() + 8
        while (not marker.exists() or (crash and not marker.with_name("prediction").exists())) and time.monotonic() < until:
            time.sleep(0.05)
        assert marker.is_file()
        assert owner.poll() is None
        pid = marker.read_text(encoding="ascii").strip()
        assert pid.isdecimal()
        assert subprocess.run(
            [str(BASH), "--noprofile", "--norc", "-c", "kill -0 " + pid],
            env=env, capture_output=True, timeout=5,
        ).returncode == 0
        contender = subprocess.run(args, input=b"", capture_output=True, env=env, timeout=5)
        assert contender.returncode == 75
        assert marker.read_text(encoding="ascii").strip() == pid
        owner.stdin.close()
        if crash:
            assert marker.with_name("prediction").is_file()
            assert subprocess.run(args, input=b"", capture_output=True, env=env, timeout=5).returncode == 75
            assert owner.poll() is None
            permit.touch()
        assert owner.wait(timeout=10) == 21
        assert not marker.parent.exists()
    finally:
        permit.touch()
        if not owner.stdin.closed:
            owner.stdin.close()
        owner.wait(timeout=10)
        owner.stdout.close()
        owner.stderr.close()
