from omegaconf import OmegaConf


def validate_opsd(config):
    cfg = config.stepopd.opsd
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if config.stepopd.algorithm != "opsd" or config.distillation.enabled:
        raise ValueError(
            "SKILL OPSD uses the actor as teacher; disable external distillation"
        )
    if not config.trainer.use_v1 or config.trainer.v1.trainer_mode != "skill_opsd":
        raise ValueError("SKILL OPSD requires the registered v1 skill_opsd trainer")
    if (
        not config.data.continuous_token.enable
        or config.data.continuous_token.model_family != "qwen3"
    ):
        raise ValueError(
            "This SKILL OPSD recipe supports Qwen3 continuous-token trajectories"
        )
    if (
        actor.strategy not in ("fsdp", "fsdp2")
        or not config.actor_rollout_ref.model.use_remove_padding
    ):
        raise ValueError(
            "SKILL OPSD requires the FSDP engine with use_remove_padding=true"
        )
    if (
        actor.use_kl_loss
        or config.algorithm.use_kl_in_reward
        or actor.entropy_coeff != 0
    ):
        raise ValueError(
            "This recipe uses only k1 PG-OPSD without additional KL or entropy loss"
        )
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError(
            "Keep GRPO bookkeeping; task rewards do not enter the OPSD loss"
        )
    if OmegaConf.select(
        config, "algorithm.rollout_correction.bypass_mode", default=False
    ):
        raise ValueError(
            "SKILL OPSD must recompute old_log_probs using the pre-update actor"
        )
    if (
        OmegaConf.select(
            config, "actor_rollout_ref.actor.router_replay.mode", default="disabled"
        )
        != "disabled"
    ):
        raise ValueError(
            "SKILL OPSD does not support replaying MoE routes across different prefixes"
        )
    if (
        not rollout.calculate_log_probs
        or rollout.temperature <= 0
        or rollout.top_p != 1
        or rollout.top_k != -1
    ):
        raise ValueError(
            "SKILL OPSD requires full-support sampling and rollout log probabilities"
        )
    if cfg.max_extra_tokens < 1:
        raise ValueError("max_extra_tokens must be positive")
    required = (
        config.data.max_prompt_length
        + config.data.max_response_length
        + cfg.max_extra_tokens
    )
    if cfg.max_teacher_sequence_length < required:
        raise ValueError(
            "Teacher context must cover the student context plus max_extra_tokens"
        )
    if rollout.log_prob_max_token_len_per_gpu < cfg.max_teacher_sequence_length:
        raise ValueError(
            "log_prob_max_token_len_per_gpu must cover max_teacher_sequence_length"
        )
    if (
        cfg.loss.loss_mode != "k1"
        or not cfg.loss.use_policy_gradient
        or cfg.loss.use_task_rewards
    ):
        raise ValueError(
            "SKILL OPSD requires loss_mode=k1, use_policy_gradient=true, "
            "use_task_rewards=false"
        )
