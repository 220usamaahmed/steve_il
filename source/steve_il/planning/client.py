"""Client of the cuRobo planner server: same API as ``ArmPlanner``, no cuRobo / warp in this process.

    with ArmPlannerClient() as planner:                      # starts the server process, waits for warm-up
        planner.set_world(prims, base_xy_yaw)
        plan = planner.plan_grasp(q, grasp_poses_base)

The server runs the *same python interpreter* in a clean environment (the Kit app changes
``LD_LIBRARY_PATH`` and friends, which would break cuRobo's torch / CUDA libraries). Its output goes
to ``logs/planner_server.log``. Always ``close()`` the client (or use ``with``) before ``os._exit``.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from steve_il import PROJECT_ROOT
from steve_il.planning.types import GraspPlan, PlanFailure
from steve_il.planning.world import Prim

LOG_PATH = PROJECT_ROOT / "logs" / "planner_server.log"
# environment variables set by Isaac Sim / Kit / conda activation that must not leak into the server
_STRIP_PREFIXES = ("CARB", "ISAAC", "OMNI", "EXP_PATH", "KIT", "RESOURCE_NAME", "LD_", "PYTHON")


def _server_env(authkey: bytes) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_STRIP_PREFIXES)}
    env["PYTHONPATH"] = str(PROJECT_ROOT / "source")
    env["STEVE_PLANNER_AUTHKEY"] = authkey.hex()
    return env


class ArmPlannerClient:
    def __init__(self, startup_timeout: float = 600.0, request_timeout: float = 120.0):
        self.request_timeout = request_timeout
        self._authkey = secrets.token_bytes(16)
        self._dir = tempfile.mkdtemp(prefix="steve_planner_")
        self._socket = os.path.join(self._dir, "sock")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(LOG_PATH, "w")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "steve_il.planning.server", self._socket],
            env=_server_env(self._authkey),
            stdout=self._log,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        self._conn = self._connect(startup_timeout)

    def _connect(self, timeout: float):
        deadline = time.time() + timeout
        conn = None
        while conn is None:
            if self._proc.poll() is not None:
                raise RuntimeError(f"planner server exited with code {self._proc.returncode}; see {LOG_PATH}")
            if time.time() > deadline:
                self.close()
                raise TimeoutError(f"planner server did not start within {timeout:.0f}s; see {LOG_PATH}")
            try:
                conn = Client(self._socket, family="AF_UNIX", authkey=self._authkey)
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(0.2)
        # the server builds and warms up the planner (kernel compile on the first run) before it says ready
        while not conn.poll(1.0):
            if self._proc.poll() is not None:
                raise RuntimeError(f"planner server died during warm-up; see {LOG_PATH}")
            if time.time() > deadline:
                self.close()
                raise TimeoutError(f"planner warm-up took longer than {timeout:.0f}s; see {LOG_PATH}")
        kind, _ = conn.recv()
        assert kind == "ready", kind
        return conn

    # -- RPC ---------------------------------------------------------------------------------
    def _call(self, method: str, *args, **kwargs):
        self._conn.send((method, args, kwargs))
        if not self._conn.poll(self.request_timeout):
            raise TimeoutError(f"planner server did not answer {method!r} within {self.request_timeout:.0f}s")
        kind, payload = self._conn.recv()
        if kind == "error":
            raise RuntimeError(f"planner server error in {method}:\n{payload}")
        return payload

    def ping(self) -> str:
        return self._call("ping")

    def set_world(self, prims: list[Prim], base_xy_yaw) -> None:
        self._call("set_world", prims, tuple(float(v) for v in base_xy_yaw))

    def attach(self, name: str, q_grasp, obj_pose_base: np.ndarray) -> None:
        self._call("attach", name, np.asarray(q_grasp), np.asarray(obj_pose_base))

    def detach(self) -> None:
        self._call("detach")

    def tcp_pose(self, q) -> np.ndarray:
        return self._call("tcp_pose", np.asarray(q))

    def plan_grasp(
        self, q_start, tcp_poses_base: np.ndarray, approach_offset: float = 0.12, lift_offset: float = 0.15
    ) -> GraspPlan | PlanFailure:
        return self._call("plan_grasp", np.asarray(q_start), np.asarray(tcp_poses_base), approach_offset, lift_offset)

    def plan_joints(self, q_start, q_goal) -> np.ndarray | PlanFailure:
        return self._call("plan_joints", np.asarray(q_start), np.asarray(q_goal))

    # -- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
            self._conn = None
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()
            self._log = None
        try:
            for f in Path(self._dir).glob("*"):
                f.unlink()
            os.rmdir(self._dir)
        except OSError:
            pass

    def __enter__(self) -> "ArmPlannerClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
