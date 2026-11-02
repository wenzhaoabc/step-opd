def prepare_records(data_dir):
    for split, folder in [
        ("train", "train"),
        ("val", "valid_seen"),
        ("test", "valid_unseen"),
    ]:
        games = sorted((data_dir / "json_2.1.1" / folder).glob("*/*/game.tw-pddl"))
        if not games:
            raise FileNotFoundError(
                f"No game.tw-pddl in {data_dir / 'json_2.1.1' / folder}"
            )
        for game in games:
            relative = game.relative_to(data_dir).as_posix()
            yield (
                split,
                {
                    "id": "alfworld:" + relative,
                    "environment": "alfworld",
                    "split": split,
                    "task": {"game_file": relative},
                },
            )
