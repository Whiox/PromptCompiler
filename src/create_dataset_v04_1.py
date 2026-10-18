
from copy import deepcopy
import json
from pathlib import Path
import random
import tiktoken
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DatasetGenerator:
    def __init__(
        self,
        source_filename: str = "raw_v03.json",
        schema_filename: str = "schema.json",
        output_dir: str = "v0.1",
        random_seed: int = 42,
    ) -> None:
        self.source_path = ROOT / "data" / source_filename
        self.schema_path = ROOT / "data" / schema_filename
        self.output_dir = ROOT / "datasets" / output_dir
        self.random_seed = random_seed

        self.raw = json.loads(self.source_path.read_text(encoding="utf-8-sig"))
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8-sig"))

    def _split(self, scenes: list) -> dict:
        held_out = max(1, len(scenes) // 6)
        splits = {
            "train": scenes[:-2 * held_out],
            "val": scenes[-2 * held_out:-held_out],
            "test": scenes[-held_out:],
        }

        return splits

    def _save_json(self, splits: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split, group in splits.items():
            rows = [
                {"scene_id": item["scene_id"], "request": request, "scene": item["scene"]}
                for item in group
                for request in item["requests"]
            ]
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (self.output_dir / f"{split}.jsonl").write_text(content, encoding="utf-8")

    def _save_bin(self, splits: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = tiktoken.get_encoding("gpt2")

        for split, group in splits.items():
            token_ids = []

            for item in group:
                scene = json.dumps(
                    item["scene"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                for request in item["requests"]:
                    text = (
                        f"### Request:\n{request}\n\n"
                        f"### Scene:\n{scene}"
                    )
                    token_ids.extend(tokenizer.encode_ordinary(text))
                    token_ids.append(tokenizer.eot_token)

            np.asarray(token_ids, dtype=np.uint16).tofile(
                self.output_dir / f"{split}.bin"
            )


    def _generate_schenes(self) -> list:
        objects = self.raw["objects"]
        relations = self.raw["relations"]
        colors = self.raw["colors"]
        materials = self.raw["materials"]

        scenes: list = []

        for obj_1 in objects:
            for obj_2 in objects:
                if obj_1 == obj_2:
                    continue

                for relation in relations:
                    scene = deepcopy(self.schema)

                    c1 = random.choice(colors)
                    c2 = random.choice(colors)

                    m1 = random.choice(materials)
                    m2 = random.choice(materials)

                    scene["objects"] = [
                        {"id": "o1", "name": obj_1, "color": c1, "material": m1},
                        {"id": "o2", "name": obj_2, "color": c2, "material": m2},
                    ]
                    scene["relations"] = [
                        {"subject": "o1", "type": relation, "object": "o2"},
                    ]

                    material_1 = "wooden" if m1 == "wood" else m1
                    material_2 = "wooden" if m2 == "wood" else m2
                    subject = f"{c1} {material_1} {obj_1}"
                    other = f"{c2} {material_2} {obj_2}"

                    requests = [
                        f"The {subject} is {relation} the {other}.",
                        f"Show the {subject} {relation} the {other}.",
                        f"Draw the {subject} {relation} the {other}.",
                        f"Depict the {subject} {relation} the {other}.",
                        f"Place the {subject} {relation} the {other}.",
                        f"Create an image of the {subject} {relation} the {other}.",
                        f"The scene contains the {subject} {relation} the {other}.",
                        f"In this scene, the {subject} is {relation} the {other}.",
                        f"The {subject} is positioned {relation} the {other}.",
                        f"The {subject} is located {relation} the {other}.",
                        f"Include the {subject} {relation} the {other} in the scene.",
                        f"I would like to see the {subject} {relation} the {other}.",
                        f"Can you show the {subject} {relation} the {other}?",
                        f"Please put the {subject} {relation} the {other}.",
                        f"The {obj_1} is {c1} and made of {m1}. It is {relation} the {other}.",
                        f"The {obj_1} is made of {m1} and colored {c1}. Show it {relation} the {other}.",
                        f"The {subject} is {relation} the {obj_2}, which is {c2} and made of {m2}.",
                        f"The {subject} is {relation} the {obj_2}. The {obj_2} is made of {m2} and colored {c2}.",
                        (
                            f"The {obj_1} is {c1} and made of {m1}. "
                            f"The {obj_2} is {c2} and made of {m2}. "
                            f"The {obj_1} is {relation} the {obj_2}."
                        ),
                        (
                            f"Show two objects: the {subject} and the {other}. "
                            f"The first is {relation} the second."
                        ),
                        (
                            f"There are two objects in the scene: the {subject} and the {other}. "
                            f"Position the {obj_1} {relation} the {obj_2}."
                        ),
                        f"The {c1} {obj_1}, made of {m1}, is {relation} the {c2} {obj_2}, made of {m2}.",
                        f"Place the {obj_1} made of {m1} {relation} the {obj_2} made of {m2}. Their colors are {c1} and {c2}, respectively.",
                        (
                            f"The {obj_1} is {relation} the {obj_2}. "
                            f"Use {c1} {m1} for the {obj_1} and {c2} {m2} for the {obj_2}."
                        ),
                        f"The {material_1} {obj_1} is {relation} the {material_2} {obj_2}. The {obj_1} is {c1}; the {obj_2} is {c2}.",
                        f"The {c1} {obj_1} is {relation} the {c2} {obj_2}. The {obj_1} is made of {m1}; the {obj_2} is made of {m2}.",
                        f"The {obj_1} ({c1}, {m1}) is {relation} the {obj_2} ({c2}, {m2}).",
                        f"Objects: {subject}; {other}. Arrangement: the {obj_1} is {relation} the {obj_2}.",
                        f"First object: {subject}. Second object: {other}. The first object is {relation} the second object.",
                        f"Show the {obj_1} {relation} the {obj_2}, with the {obj_1} in {c1} {m1} and the {obj_2} in {c2} {m2}.",
                        (
                            f"Imagine the {subject} and the {other}. "
                            f"The position of the {obj_1} should be {relation} the {obj_2}."
                        ),
                        f"Arrange the {subject} and the {other} so that the {obj_1} is {relation} the {obj_2}.",
                        f"I need an image containing the {subject} and the {other}, with the {obj_1} {relation} the {obj_2}.",
                        f"The scene should show the {subject} {relation} the {other}, with both objects visible.",
                    ]

                    scenes.append({
                        "scene_id": f"scene_{len(scenes) + 1:03d}",
                        "scene": scene,
                        "requests": requests,
                    })

        random.Random(self.random_seed).shuffle(scenes)

        return scenes

    def run(self, output: str = "bin") -> None:
        scenes = self._generate_schenes()
        print(f"total scenes: {len(scenes)}")
        splits = self._split(scenes)
        if output == "bin":
            self._save_bin(splits)
        else:
            self._save_json(splits)


if __name__ == "__main__":
    generator = DatasetGenerator(
        source_filename="raw_v04.json",
        schema_filename="schema.json",
        output_dir="v0.4_1",
    )
    generator.run()
    print("Dataset generation completed.")
