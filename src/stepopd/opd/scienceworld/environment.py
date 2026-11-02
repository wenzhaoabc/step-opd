from ..episode import State, parse_command
from .protocol import HISTORY_LENGTH, MEMORY_FORMAT, TEMPLATE, TEMPLATE_NO_HIS


class ScienceWorld:
    max_steps = 30

    def __init__(self, task):
        self.task_name = task["task_name"]
        self.variation = int(task["variation"])
        self.env = None
        self.best_score = 0
        self.final_score = 0
        self.state = State("")
        self.task_description = ""
        self.observation = ""
        self.history = []
        self.turn = 0

    def reset(self):
        from scienceworld import ScienceWorldEnv
        from scienceworld.constants import JAR_PATH

        self.env = ScienceWorldEnv("", JAR_PATH, envStepLimit=100)
        self.env.load(self.task_name, self.variation, "easy", generateGoldPath=False)
        observation, info = self.env.reset()
        self.best_score = self.final_score = int(info["score"])
        self.observation = observation
        self.task_description = self.env.get_task_description().strip()
        self.history, self.turn = [], 0
        return self._observe(False)

    def _render(self):
        actions = [action for action in self.env.get_possible_actions() if action != "help"]
        objects = self.env.get_possible_objects()
        fields = {
            "task_description": self.task_description,
            "current_observation": self.observation.strip(),
            "action_templates": ", ".join(f"'{s}'" for s in actions),
            "objects": ", ".join(f"'{s}'" for s in objects),
        }
        if len(self.history) < HISTORY_LENGTH:
            return TEMPLATE_NO_HIS.format(**fields)
        return TEMPLATE.format(
            step_count=self.turn,
            history_length=min(HISTORY_LENGTH, len(self.history)),
            action_history="\n".join(self.history[-HISTORY_LENGTH:]),
            current_step=self.turn + 1,
            **fields,
        )

    def _observe(self, done):
        success = self.final_score == 100
        reason = "success" if success else "environment_done" if done else ""
        self.state = State(
            self._render(),
            self.best_score / 100.0,
            bool(done),
            success,
            reason,
            final_score=self.final_score,
        )
        return self.state

    def step(self, response):
        try:
            _, action = parse_command(response, {"action"})
        except ValueError as error:
            return State(
                str(error) + "\n" + self.state.text,
                self.state.score,
                invalid_action=True,
                final_score=self.final_score,
            )
        self.history.append(
            MEMORY_FORMAT.format(step_num=self.turn + 1, obs=self.observation.strip(), act=action)
        )
        observation, _, done, info = self.env.step(action)
        self.observation = observation
        self.turn += 1
        self.final_score = int(info["score"])
        self.best_score = max(self.best_score, self.final_score)
        return self._observe(done)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
