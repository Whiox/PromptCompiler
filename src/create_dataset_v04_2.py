
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
        missing_attribute_probability: float = 0.3,
    ) -> None:
        self.source_path = ROOT / "data" / source_filename
        self.schema_path = ROOT / "data" / schema_filename
        self.output_dir = ROOT / "datasets" / output_dir
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)
        if not 0 <= missing_attribute_probability <= 1:
            raise ValueError("missing_attribute_probability must be between 0 and 1")
        self.missing_attribute_probability = missing_attribute_probability

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


    def _choose_attribute(self, values: list) -> str | None:
        if not values or self.rng.random() < self.missing_attribute_probability:
            return None
        return self.rng.choice(values)

    def _describe_object(self, name: str, color: str | None, material: str | None) -> str:
        adjective = "wooden" if material == "wood" else material
        return " ".join(value for value in (color, adjective, name) if value)

    def _describe_properties(self, name: str, color: str | None, material: str | None) -> str:
        properties = []
        if color:
            properties.append(color)
        if material:
            properties.append(f"made of {material}")
        if not properties:
            return ""
        return f"The {name} is {' and '.join(properties)}."

    def _make_requests(
        self,
        obj_1: str, c1: str | None, m1: str | None,
        relation: str,
        obj_2: str, c2: str | None, m2: str | None,
    ) -> list[str]:
        subject = self._describe_object(obj_1, c1, m1)
        other = self._describe_object(obj_2, c2, m2)
        properties_1 = self._describe_properties(obj_1, c1, m1)
        properties_2 = self._describe_properties(obj_2, c2, m2)
        details = " ".join(value for value in (properties_1, properties_2) if value)
        position = f"The {obj_1} is {relation} the {obj_2}."
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
            f"Show the {obj_1} {relation} the {other}. {properties_1}",
            f"The {subject} is {relation} the {obj_2}. {properties_2}",
            f"{position} {details}",
            f"The scene has the {obj_1} and the {obj_2}. {details} {position}",
            f"Show two objects: the {subject} and the {other}. The first is {relation} the second.",
            f"There are two objects: the {subject} and the {other}. Position the {obj_1} {relation} the {obj_2}.",
            f"Objects: {subject}; {other}. Arrangement: {position}",
            f"First object: {subject}. Second object: {other}. The first object is {relation} the second object.",
            f"Imagine the {subject} and the {other}. The position of the {obj_1} should be {relation} the {obj_2}.",
            f"Arrange the {subject} and the {other} so that the {obj_1} is {relation} the {obj_2}.",
            f"I need an image containing the {subject} and the {other}, with the {obj_1} {relation} the {obj_2}.",
            f"The scene should show the {subject} {relation} the {other}, with both objects visible.",
            f"Show the {obj_1} and the {obj_2}. {position} {details}",
            f"Include the {obj_1} and the {obj_2}. {details} Position the {obj_1} {relation} the {obj_2}.",
            f"Show the {subject} and the {other}, with the first object {relation} the second.",
            f"Arrange two objects: the {subject} and the {other}. Put the first {relation} the second.",
            f"The {subject} should appear {relation} the {other}.",
            f"Create a scene with the {subject} {relation} the {other}.",
            f"Put the {subject} {relation} the {other} in the image.",
            f"Show the {obj_1} {relation} the {obj_2}. {details}",
        ]
        # Remove extra whitespace and duplicates when property clauses are absent.
        return list(dict.fromkeys(" ".join(request.split()) for request in requests))

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

                    c1 = self._choose_attribute(colors)
                    c2 = self._choose_attribute(colors)

                    m1 = self._choose_attribute(materials)
                    m2 = self._choose_attribute(materials)

                    scene["objects"] = [
                        {"id": "o1", "name": obj_1, "color": c1, "material": m1},
                        {"id": "o2", "name": obj_2, "color": c2, "material": m2},
                    ]
                    scene["relations"] = [
                        {"subject": "o1", "type": relation, "object": "o2"},
                    ]

                    requests = self._make_requests(obj_1, c1, m1, relation, obj_2, c2, m2)

                    scenes.append({
                        "scene_id": f"scene_{len(scenes) + 1:03d}",
                        "scene": scene,
                        "requests": requests,
                    })

        self.rng.shuffle(scenes)

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
        output_dir="v0.4_2",
    )
    generator.run()
    print("Dataset generation completed.")
