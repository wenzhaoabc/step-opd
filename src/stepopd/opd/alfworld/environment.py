import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..episode import State, parse_command
from .protocol import HISTORY_LENGTH, MEMORY_FORMAT, TASK_PREFIX, TEMPLATE, TEMPLATE_NO_HIS

_REQUEST_TIMEOUT = 60
_CLOSE_TIMEOUT = 2


class _TextWorldProcess:
    def __init__(self, game_file, max_steps):
        self._lock = threading.Lock()
        self.socket, child = socket.socketpair()
        self._tempdir = None
        try:
            self._tempdir = tempfile.TemporaryDirectory(prefix="stepopd-alfworld-")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).with_name("native.py")),
                    str(child.fileno()),
                    str(game_file),
                    str(max_steps),
                    str(os.getpid()),
                ],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONFAULTHANDLER": "1", "TMPDIR": self._tempdir.name},
            )
        except BaseException:
            self.socket.close()
            if self._tempdir is not None:
                self._tempdir.cleanup()
            raise
        finally:
            child.close()

    def _request(self, command, **payload):
        with self._lock:
            channel = self.socket
        if channel is None:
            raise RuntimeError("ALFWorld subprocess is closed")
        deadline = time.monotonic() + _REQUEST_TIMEOUT
        try:
            channel.settimeout(_REQUEST_TIMEOUT)
            channel.sendall(json.dumps({"command": command, **payload}).encode() + b"\n")
            data = bytearray()
            while not data.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ALFWorld response deadline exceeded")
                channel.settimeout(min(remaining, 0.1))
                try:
                    chunk = channel.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    raise EOFError("ALFWorld subprocess closed its response channel")
                data.extend(chunk)
            response = json.loads(data)
            if "error" in response:
                raise RuntimeError(f"ALFWorld subprocess failed:\n{response['error']}")
            return response["result"]
        except BaseException as error:
            self.close()
            if isinstance(error, TimeoutError):
                raise TimeoutError(f"ALFWorld {command} timed out after {_REQUEST_TIMEOUT}s") from error
            if isinstance(error, (OSError, EOFError)):
                raise RuntimeError(
                    f"ALFWorld subprocess pid={self.process.pid} exited or disconnected "
                    f"(exitcode={self.process.returncode}) during {command}"
                ) from error
            raise

    def reset(self):
        return self._request("reset")

    def step(self, action):
        return self._request("step", action=action)

    def close(self):
        with self._lock:
            if self.socket is None:
                return
            channel, self.socket = self.socket, None
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            channel.close()
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=_CLOSE_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=_CLOSE_TIMEOUT)
            self._tempdir.cleanup()


class ALFWorld:
    max_steps = 30

    def __init__(self, task, data_dir):
        self.game_file = (data_dir / task["game_file"]).resolve()
        if not self.game_file.is_file():
            raise FileNotFoundError(self.game_file)
        self.env = None
        self._lock = threading.RLock()
        self._closed = False
        self.state = State("")
        self.task_description = ""
        self.observation = ""
        self.actions = []
        self.history = []
        self.turn = 0

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("ALFWorld is closed; create a new environment for another episode")
            if self.env is not None:
                self.env.close()
                self.env = None
            self.env = _TextWorldProcess(self.game_file, self.max_steps)
            env = self.env
        try:
            observation, info = env.reset()
            self.observation = observation
            self.actions = info["admissible_commands"]
            start = observation.find(TASK_PREFIX)
            if start == -1:
                raise ValueError("Task description not found in the initial observation.")
            self.task_description = observation[start + len(TASK_PREFIX) :].strip()
            self.history, self.turn = [], 0
            self.state = State(self._render())
            return self.state
        except BaseException:
            self.close()
            raise

    def _render(self):
        admissible = "\n ".join(f"'{s}'" for s in self.actions if s != "help")
        if len(self.history) < HISTORY_LENGTH:
            return TEMPLATE_NO_HIS.format(
                current_observation=self.observation.strip(),
                admissible_actions=admissible,
            )
        return TEMPLATE.format(
            task_description=self.task_description,
            step_count=self.turn,
            history_length=min(HISTORY_LENGTH, len(self.history)),
            action_history="\n".join(self.history[-HISTORY_LENGTH:]),
            current_step=self.turn + 1,
            current_observation=self.observation.strip(),
            admissible_actions=admissible,
        )

    def step(self, response):
        try:
            _, action = parse_command(response, {"action"})
            if action not in self.actions:
                raise ValueError("Choose an action from the admissible actions.")
        except ValueError as error:
            return State(str(error) + "\n" + self.state.text, self.state.score, invalid_action=True)
        self.history.append(MEMORY_FORMAT.format(step_num=self.turn + 1, obs=self.observation.strip(), act=action))
        observation, _, done, info = self.env.step(action)
        self.observation = observation
        self.actions = info["admissible_commands"]
        self.turn += 1
        success = bool(info["won"])
        self.state = State(
            self._render(),
            float(success),
            bool(done),
            success,
            "success" if success else "environment_done" if done else "",
        )
        return self.state

    def close(self):
        with self._lock:
            self._closed = True
            env, self.env = self.env, None
            if env is not None:
                env.close()
