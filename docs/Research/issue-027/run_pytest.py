import os
import socket
import sys
from pathlib import Path

assert os.environ.get("ISSUE27_SOCKET_DENIAL") == "active"
with socket.socket(socket.AF_INET) as connection:
    try:
        connection.connect(("127.0.0.1", 1234))
    except PermissionError as error:
        assert "issue #27 audit" in str(error)
    else:
        raise AssertionError("network guard failed")
try:
    socket.getaddrinfo("localhost", 1234)
except PermissionError as error:
    assert "issue #27 audit" in str(error)
else:
    raise AssertionError("DNS guard failed")
print("TCP and DNS denial self-check passed", flush=True)
import pytest
sys.path.insert(0, str(Path.cwd()))
sys.exit(pytest.main(sys.argv[1:]))
