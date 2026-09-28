"""Minimal Python client for the rtier controller API (internal/ctrl over internal/frame).

Frames are little endian: magic "RTF1" | length | type | flags | status | req_id | epoch |
body | "FEND"; control bodies are JSON envelopes {"m": method, "err": ..., "body": ...}.

    from rtier_ctl import Controller
    c = Controller("127.0.0.1:7101")
    c.call("Status")
    c.call("Rescale", {"data_nodes": 3})
"""

import json
import socket
import struct

MAGIC_START = 0x31465452
MAGIC_END = 0x444E4546
TYPE_REQUEST, TYPE_RESPONSE, TYPE_NOTIFY = 1, 2, 3


class RemoteError(Exception):
    pass


class Controller:
    def __init__(self, addr: str, timeout: float = 3600.0):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.next_id = 0

    def close(self):
        self.sock.close()

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("controller closed the connection")
            buf += chunk
        return buf

    def _send(self, typ: int, req_id: int, env: dict):
        body = json.dumps(env).encode()
        length = 20 + len(body) + 4
        head = struct.pack("<IIBBHQQ", MAGIC_START, length, typ, 0, 0, req_id, 0)
        self.sock.sendall(head + body + struct.pack("<I", MAGIC_END))

    def _read(self):
        magic, length = struct.unpack("<II", self._recv_exact(8))
        if magic != MAGIC_START:
            raise ConnectionError("bad frame magic")
        rest = self._recv_exact(length)
        typ, _flags, _status, req_id, _epoch = struct.unpack("<BBHQQ", rest[:20])
        if struct.unpack("<I", rest[-4:])[0] != MAGIC_END:
            raise ConnectionError("bad frame trailer")
        return typ, req_id, json.loads(rest[20:-4] or b"{}")

    def call(self, method: str, body=None):
        self.next_id += 1
        rid = self.next_id
        self._send(TYPE_REQUEST, rid, {"m": method, "body": body})
        while True:
            typ, req_id, env = self._read()
            if typ == TYPE_RESPONSE and req_id == rid:
                if env.get("err"):
                    raise RemoteError(f"{method}: {env['err']}")
                return env.get("body")


if __name__ == "__main__":
    import sys

    c = Controller(sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:7101")
    print(json.dumps(c.call(sys.argv[2] if len(sys.argv) > 2 else "Status"), indent=2))
