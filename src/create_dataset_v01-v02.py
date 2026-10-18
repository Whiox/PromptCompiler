
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
        source_filename: str = "raw_v01.json",
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

        self.rng = random.Random(self.random_seed)

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

        scenes: list = []

        for obj_1 in objects:
            for obj_2 in objects:
                if obj_1 == obj_2:
                    continue

                for relation in relations:
                    scene = deepcopy(self.schema)

                    scene["objects"] = [
                        {"id": "o1", "name": obj_1},
                        {"id": "o2", "name": obj_2},
                    ]
                    scene["relations"] = [
                        {"subject": "o1", "type": relation, "object": "o2"},
                    ]

                    requests = [
                        f"The {obj_1} is {relation} the {obj_2}.",
                        f"Show the {obj_1} {relation} the {obj_2}.",
                    ]

                    scenes.append({
                        "scene_id": f"scene_{len(scenes) + 1:03d}",
                        "scene": scene,
                        "requests": requests,
                    })

        self.rng.shuffle(scenes)

        return scenes

    def run(self, output: str = "bin") -> None:
        scenes = self._generate_schenes()
        splits = self._split(scenes)
        if output == "bin":
            self._save_bin(splits)
        else:
            self._save_json(splits)


if __name__ == "__main__":
    generator = DatasetGenerator(
        source_filename="raw_v01.json",
        schema_filename="schema.json",
        output_dir="v0.1",
    )
    generator.run()
    print("Dataset generation completed.")
