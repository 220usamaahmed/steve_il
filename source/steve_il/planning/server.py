"""Process that owns the cuRobo ``ArmPlanner`` and answers requests from ``ArmPlannerClient``.

Started by the client (never by hand); one client per server. Protocol over a
``multiprocessing.connection`` Unix socket: the client sends ``(method, args, kwargs)`` and gets
``("ok", result)`` or ``("error", traceback_text)`` back. After the planner has been built and
warmed up the server sends ``("ready", None)``. It exits when the client disconnects.

    python -m steve_il.planning.server SOCKET_PATH      # authkey in env STEVE_PLANNER_AUTHKEY (hex)
"""

from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.connection import Listener

# methods a client may call (everything else is refused)
METHODS = frozenset({"set_world", "attach", "detach", "plan_grasp", "plan_joints", "tcp_pose", "ping"})


def serve(socket_path: str, authkey: bytes) -> None:
    from steve_il.planning.arm_planner import ArmPlanner

    planner = ArmPlanner()
    with Listener(socket_path, family="AF_UNIX", authkey=authkey) as listener:
        conn = listener.accept()
        conn.send(("ready", None))
        while True:
            try:
                method, args, kwargs = conn.recv()
            except (EOFError, ConnectionError):
                return
            try:
                if method not in METHODS:
                    raise ValueError(f"unknown method {method!r}")
                result = "pong" if method == "ping" else getattr(planner, method)(*args, **kwargs)
                conn.send(("ok", result))
            except Exception:  # noqa: BLE001 - reported to the client, the server keeps running
                conn.send(("error", traceback.format_exc()))


if __name__ == "__main__":
    serve(sys.argv[1], bytes.fromhex(os.environ["STEVE_PLANNER_AUTHKEY"]))
