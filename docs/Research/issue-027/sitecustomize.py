import os
import socket

def deny_network(*args, **kwargs):
    raise PermissionError("issue #27 audit: networking disabled")

for name in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, name):
        setattr(socket.socket, name, deny_network)

for name in ("create_connection", "getaddrinfo", "getnameinfo",
             "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
    setattr(socket, name, deny_network)

os.environ["ISSUE27_SOCKET_DENIAL"] = "active"
