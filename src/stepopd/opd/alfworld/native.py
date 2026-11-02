import json
import os
import socket
import sys
import threading
import time
import traceback


def _watch_parent(parent_pid):
    while True:
        if os.getppid() != parent_pid:
            os._exit(1)
        time.sleep(1)


def main():
    fd, game_file, max_steps, parent_pid = sys.argv[1:]
    threading.Thread(target=_watch_parent, args=(int(parent_pid),), daemon=True).start()
    with socket.socket(fileno=int(fd)) as channel, channel.makefile("rwb") as stream:
        env = None
        try:
            import textworld
            from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
            from textworld.envs.pddl.pddl import PddlEnv
            from textworld.envs.wrappers import Filter, Limit

            for line in stream:
                request = json.loads(line)
                if request["command"] == "reset":
                    if env is not None:
                        raise RuntimeError("A native process may initialize only one episode")
                    infos = textworld.EnvInfos(admissible_commands=True, won=True)
                    env = Filter(AlfredDemangler(Limit(PddlEnv(infos), max_episode_steps=int(max_steps))))
                    env.load(game_file)
                    result = env.reset()
                elif request["command"] == "step":
                    if env is None:
                        raise RuntimeError("Environment must be reset before step")
                    result = env.step(request["action"])
                else:
                    raise ValueError(f"Unknown environment command: {request['command']}")
                stream.write(json.dumps({"result": result}).encode() + b"\n")
                stream.flush()
        except Exception:
            stream.write(json.dumps({"error": traceback.format_exc()}).encode() + b"\n")
            stream.flush()


if __name__ == "__main__":
    main()
