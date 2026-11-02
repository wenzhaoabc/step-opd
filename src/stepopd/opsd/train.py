import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from stepopd.common import write_json
from stepopd.opd.train import run_training

from .config import validate_opsd
from .skills import SkillBank


def prepare_skills(config):
    validate_opsd(config)
    cfg = config.stepopd.opsd
    bank = SkillBank(cfg.skill_root, config.stepopd.environment, cfg.default_category)
    cfg.skill_root = str(bank.root)
    cfg.manifest_sha256 = bank.sha256
    output = Path(config.trainer.default_local_dir).expanduser().resolve()
    manifest_path = output / "skills_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text())["sha256"] != bank.sha256
    ):
        raise ValueError(
            "SKILL bank differs from this snapshot; use a new experiment directory"
        )
    write_json(
        manifest_path, {**bank.manifest, "sha256": bank.sha256, "root": str(bank.root)}
    )
    for doc in bank.documents.values():
        snapshot = output / "skills" / f"{doc.sha256}.md"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(doc.text, encoding="utf-8")


def launch(config):
    from verl.trainer.main_ppo import run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    from .trainer import SkillOPSDTaskRunner

    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, SkillOPSDTaskRunner)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    prepare_skills(config)
    run_training(config, launch=launch)


if __name__ == "__main__":
    main()
