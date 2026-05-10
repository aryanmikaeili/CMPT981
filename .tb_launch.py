"""Launcher that fixes 'localhost' DNS lookup before starting TensorBoard.

This box's /etc/hosts has no `127.0.0.1 localhost` entry, so TensorBoard's
internal `connect_ex(("localhost", port))` raises gaierror.
Patch socket.socket.connect/connect_ex to rewrite 'localhost' -> '127.0.0.1'.
"""
import socket


def _rewrite(addr):
    if isinstance(addr, tuple) and len(addr) >= 1 and addr[0] in (
        "localhost",
        "localhost.localdomain",
    ):
        return ("127.0.0.1",) + tuple(addr[1:])
    return addr


_orig_connect_ex = socket.socket.connect_ex
_orig_connect = socket.socket.connect


def _patched_connect_ex(self, addr):
    return _orig_connect_ex(self, _rewrite(addr))


def _patched_connect(self, addr):
    return _orig_connect(self, _rewrite(addr))


socket.socket.connect_ex = _patched_connect_ex
socket.socket.connect = _patched_connect

from tensorboard.main import run_main  # noqa: E402

raise SystemExit(run_main())
