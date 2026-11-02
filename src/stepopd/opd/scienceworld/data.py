import random

TRAIN_TASKS = {
    "boil": 15,
    "melt": 15,
    "change-the-state-of-matter-of": 15,
    "use-thermometer": 270,
    "measure-melting-point-known-substance": 218,
    "power-component": 10,
    "test-conductivity": 450,
    "find-living-thing": 150,
    "find-plant": 150,
    "grow-plant": 63,
    "chemistry-mix": 16,
    "chemistry-mix-paint-secondary-color": 18,
    "lifespan-shortest-lived": 62,
    "identify-life-stages-2": 5,
    "inclined-plane-determine-angle": 84,
    "inclined-plane-friction-named-surfaces": 693,
    "mendelian-genetics-known-plant": 60,
}
TEST_TASKS = {
    "freeze": 15,
    "measure-melting-point-unknown-substance": 150,
    "power-component-renewable-vs-nonrenewable-energy": 10,
    "test-conductivity-of-unknown-substances": 300,
    "find-non-living-thing": 150,
    "find-animal": 150,
    "grow-fruit": 63,
    "chemistry-mix-paint-tertiary-color": 18,
    "lifespan-longest-lived": 62,
    "lifespan-longest-lived-then-shortest-lived": 62,
    "identify-life-stages-1": 7,
    "inclined-plane-friction-unnamed-surfaces": 81,
    "mendelian-genetics-unknown-plant": 240,
}


def prepare_records():
    for split, tasks in [
        ("train", TRAIN_TASKS),
        ("val", TRAIN_TASKS),
        ("test", TEST_TASKS),
    ]:
        rows = []
        for name, count in tasks.items():
            variations = range(count, count + 5) if split == "val" else range(count)
            for variation in variations:
                rows.append(
                    {
                        "id": f"scienceworld:{name}:{variation}",
                        "environment": "scienceworld",
                        "split": split,
                        "task": {"task_name": name, "variation": variation},
                    }
                )
        random.Random(42).shuffle(rows)
        for row in rows:
            yield split, row
