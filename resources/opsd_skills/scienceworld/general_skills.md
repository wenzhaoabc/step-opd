# ScienceWorld general skills

These are manually authored starter strategies for this project's ScienceWorld interface.
They are not skills extracted from AgentOPSD or from a set of successful trajectories.

- Translate the task description into observable subgoals and track which have been verified.
- Use only action templates and object names available in the current observation. Do not invent tools or objects.
- Inspect an unfamiliar object or location when the resulting observation is needed to select the next action.
- Distinguish what was observed from what is only a hypothesis. For a measurement task, obtain the requested measurement before drawing a conclusion.
- Verify action outcomes in the next observation. An attempted action is not evidence that a subgoal was completed.
- After an invalid action, check the command and object names before retrying. Avoid repeating an unchanged action that has already failed.
- Preserve the environment's requested output format and take one environment action at a time.
