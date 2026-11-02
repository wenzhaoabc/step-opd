import argparse
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from threading import Event, Thread

import psutil


class Stage:
    def __init__(self, output, reader, grace_seconds):
        self.output = output
        self.reader = reader
        self.grace = grace_seconds
        self.owner = psutil.Process()
        self.owned = set()
        self.driver = None
        self.signals = []
        self.started = time.monotonic()
        self.finished = Event()
        self.inspection_error = None

    def event(self, name, **fields):
        record = {"event": name, "elapsed": round(time.monotonic() - self.started, 3)}
        self.output.write(("\n[stage] " + json.dumps(record | fields) + "\n").encode())

    def mirror(self):
        try:
            while True:
                chunk = self.reader.read(65536)
                if not chunk:
                    if self.finished.is_set():
                        return
                    self.finished.wait(0.05)
                    continue
                while chunk:
                    try:
                        chunk = chunk[os.write(sys.stdout.fileno(), chunk) :]
                    except BlockingIOError:
                        if self.finished.wait(0.05):
                            return
        except (OSError, ValueError):
            return

    def descendants(self):
        self.inspection_error = None
        try:
            self.owned.update(self.owner.children(recursive=True))
        except (psutil.Error, OSError) as error:
            self.inspection_error = error
        alive = set()
        for process in self.owned:
            try:
                if not process.is_running():
                    continue
                if process.status() == psutil.STATUS_ZOMBIE:
                    if self.driver is None or process.pid != self.driver.pid:
                        with suppress(ChildProcessError):
                            os.waitpid(process.pid, os.WNOHANG)
                    continue
                alive.add(process)
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, OSError) as error:
                self.inspection_error = error
                alive.add(process)
        self.owned = alive
        return alive

    def send(self, processes, signum):
        for process in processes:
            try:
                process.send_signal(signum)
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as error:
                with suppress(OSError):
                    self.event("signal_failed", pid=process.pid, error=str(error))

    def run(self, command, timeout):
        self.driver = subprocess.Popen(
            command,
            env=os.environ | {"STEPOPD_MANAGED_STAGE": "1"},
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=self.output,
            stderr=subprocess.STDOUT,
        )
        with suppress(psutil.NoSuchProcess):
            self.owned.add(psutil.Process(self.driver.pid))
        self.event(
            "start",
            pid=self.driver.pid,
            command=command,
            orphan_adoption=sys.platform == "linux",
            grace_seconds=self.grace,
        )
        phase, deadline, result = "running", math.inf, None
        sent = set()
        last_progress = self.started
        empty_scans = 0
        while True:
            driver_rc = self.driver.poll()
            alive = self.descendants()
            if self.inspection_error is not None:
                raise self.inspection_error
            now = time.monotonic()
            if self.signals and result is None:
                result = 128 + self.signals[0]
            if phase == "running":
                if self.signals or (
                    timeout is not None and now - self.started >= timeout
                ):
                    signum = self.signals[0] if self.signals else signal.SIGTERM
                    if result is None:
                        result = 124
                    self.send({p for p in alive if p.pid == self.driver.pid}, signum)
                    phase, deadline = "driver_stop", now + self.grace
                    self.event(phase, signal=int(signum), deadline_seconds=self.grace)
                elif driver_rc is not None:
                    phase, deadline = "drain", now + self.grace
                    self.event(
                        phase, driver_returncode=driver_rc, deadline_seconds=self.grace
                    )
            if phase != "running":
                empty_scans = (
                    empty_scans + 1 if not alive and driver_rc is not None else 0
                )
                if empty_scans >= 2:
                    break
                if len(self.signals) >= 2 and phase != "kill":
                    phase, deadline = "kill", now + 5
                    sent.clear()
                    self.event(phase, reason="repeated_signal", deadline_seconds=5)
                elif now >= deadline:
                    if phase == "kill":
                        self.event(
                            "cleanup_failed",
                            remaining_pids=sorted(p.pid for p in alive),
                        )
                        return 125
                    phase = "kill" if phase == "terminate" else "terminate"
                    deadline = now + (5 if phase == "kill" else self.grace)
                    sent.clear()
                    self.event(
                        phase,
                        deadline_seconds=round(deadline - now, 3),
                        remaining_pids=sorted(p.pid for p in alive),
                    )
                if phase in ("terminate", "kill"):
                    self.send(
                        alive - sent,
                        signal.SIGKILL if phase == "kill" else signal.SIGTERM,
                    )
                    sent.update(alive)
                if now - last_progress >= 5:
                    self.event(
                        "cleanup_wait",
                        phase=phase,
                        remaining_processes=len(alive),
                        deadline_seconds=round(max(0, deadline - now), 1),
                    )
                    last_progress = now
            time.sleep(0.05)
        driver_rc = self.driver.wait()
        result = (
            result
            if result is not None
            else (driver_rc if driver_rc >= 0 else 128 - driver_rc)
        )
        self.event(
            "complete", exit_code=result, driver_returncode=driver_rc, remaining_pids=[]
        )
        return result

    def emergency_cleanup(self):
        for signum, seconds in ((signal.SIGTERM, self.grace), (signal.SIGKILL, 5)):
            deadline = time.monotonic() + seconds
            sent = set()
            while time.monotonic() < deadline:
                self.driver.poll()
                alive = self.descendants()
                self.send(alive - sent, signum)
                sent.update(alive)
                if not alive and self.inspection_error is None:
                    return
                if signum == signal.SIGTERM and len(self.signals) >= 2:
                    break
                time.sleep(0.05)
        with suppress(OSError):
            self.event(
                "cleanup_failed",
                remaining_pids=sorted(p.pid for p in self.owned),
                inspection_error=repr(self.inspection_error),
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=60,
        help="natural drain / driver stop / descendant TERM grace (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="training time limit in seconds, excluding cleanup",
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required after the log path")
    for value in (args.grace_seconds, args.timeout):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("timeouts must be finite positive seconds")
    if os.name != "posix":
        parser.error("stage requires POSIX processes and signals")
    if "STEPOPD_MANAGED_STAGE" in os.environ:
        parser.error("nested stages are not supported; wrap the outer launcher only")
    with ExitStack() as stack:
        try:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(args.log.open("ab", buffering=0))
            reader = stack.enter_context(args.log.open("rb", buffering=0))
        except OSError as error:
            parser.exit(125, f"cannot open stage log: {error}\n")
        reader.seek(0, os.SEEK_END)
        stage = Stage(output, reader, args.grace_seconds)
        handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for signum in handlers:
            signal.signal(signum, lambda number, frame: stage.signals.append(number))
        mirror = Thread(target=stage.mirror, daemon=True)
        mirror.start()
        result = 125
        try:
            if sys.platform == "linux":
                libc = ctypes.CDLL(None, use_errno=True)
                libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
                if libc.prctl(36, 1, 0, 0, 0) != 0:
                    raise OSError(ctypes.get_errno(), "cannot enable child subreaper")
            stage.descendants()
            if stage.inspection_error is not None:
                raise stage.inspection_error
            if sys.platform != "linux":
                stage.event(
                    "limited_platform",
                    message="no orphan adoption; detached orphans require Linux",
                )
            result = stage.run(args.command, args.timeout)
        except Exception as error:
            with suppress(OSError):
                stage.event("supervisor_failed", error=repr(error))
            if stage.driver is not None:
                try:
                    stage.emergency_cleanup()
                except Exception as cleanup_error:
                    with suppress(OSError):
                        stage.event("cleanup_failed", error=repr(cleanup_error))
        finally:
            stage.finished.set()
            mirror.join(timeout=1)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
    sys.exit(result)


if __name__ == "__main__":
    main()
