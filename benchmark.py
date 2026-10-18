import json
from pathlib import Path

from inference import Model
import tqdm


MODELS_TO_COMPARE = [
    # "nanogpt_v0.1_seed42",
    # "nanogpt_v0.2_seed42",
    "nanogpt_v0.3_seed42",
    "nanogpt_v0.4_seed42",
    "nanogpt_v0.4_1_seed42",
    "nanogpt_v0.4_2_seed42",
]

TESTS = json.loads(Path(__file__).with_name("tests.json").read_text(encoding="utf-8"))
TOTAL_OBJECTS = sum(len(test["expected"]["objects"]) for test in TESTS)

class Benchmark:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.model = None

        self.json_validity: int = 0
        self.valid_objects: int = 0
        self.valid_scenes: int = 0

    def load_model(self):
        self.model = Model(self.model_name)

    def prepare(self) -> None:
        self.json_validity: int = 0
        self.valid_objects: int = 0
        self.valid_scenes: int = 0

        self.load_model()

    def _valid_json(self, text: str) -> dict | bool:
        try:
            response = json.loads(text)
            self.json_validity += 1
            return response
        except json.decoder.JSONDecodeError:
            return False

    def _count_valid_objects(self, pred_json: dict, test_json: dict) -> int:
        if not isinstance(pred_json, dict) or not isinstance(pred_json.get("objects"), list):
            return 0

        remaining = list(test_json["objects"])
        count = 0
        for obj_pred in pred_json["objects"]:
            if not isinstance(obj_pred, dict):
                continue
            for obj_test in remaining:
                fields = (obj_pred.keys() | obj_test.keys()) - {"id"}
                if all(obj_pred.get(field) == obj_test.get(field) for field in fields):
                    count += 1
                    remaining.remove(obj_test)
                    break
        return count

    def _valid_json_structure(self, pred_json: dict, test_json: dict) -> bool:
        if not isinstance(pred_json, dict) or pred_json.keys() != test_json.keys():
            return False

        objects = pred_json.get("objects")
        expected_objects = test_json["objects"]
        if not isinstance(objects, list) or len(objects) != len(expected_objects):
            return False

        remaining = list(expected_objects)
        id_mapping = {}
        for obj_pred in objects:
            if not isinstance(obj_pred, dict):
                return False
            object_id = obj_pred.get("id")
            if not isinstance(object_id, str) or not object_id or object_id in id_mapping:
                return False

            for obj_test in remaining:
                fields = (obj_pred.keys() | obj_test.keys()) - {"id"}
                if all(obj_pred.get(field) == obj_test.get(field) for field in fields):
                    id_mapping[object_id] = obj_test["id"]
                    remaining.remove(obj_test)
                    break
            else:
                return False

        relations = pred_json.get("relations")
        if not isinstance(relations, list) or len(relations) != len(test_json["relations"]):
            return False

        expected_relations = list(test_json["relations"])
        for relation in relations:
            if not isinstance(relation, dict) or set(relation) != {"subject", "type", "object"}:
                return False
            subject, obj = relation["subject"], relation["object"]
            if not isinstance(subject, str) or not isinstance(obj, str):
                return False
            if subject not in id_mapping or obj not in id_mapping:
                return False
            normalized = {
                "subject": id_mapping[subject],
                "type": relation["type"],
                "object": id_mapping[obj],
            }
            if normalized not in expected_relations:
                return False
            expected_relations.remove(normalized)

        for field in test_json.keys() - {"objects", "relations"}:
            if pred_json[field] != test_json[field]:
                return False

        return True


    def start(self) -> None:
        for test in tqdm.tqdm(TESTS):
            response = self.model.generate_scene(test["request"])

            response_json = self._valid_json(response)

            if not response_json:
                continue

            self.valid_objects += self._count_valid_objects(response_json, test["expected"])
            if self._valid_json_structure(response_json, test["expected"]):
                self.valid_scenes += 1


def main() -> None:
    for model_name in MODELS_TO_COMPARE:
        benchmark = Benchmark(model_name)
        benchmark.prepare()
        benchmark.start()
        print(f"{model_name}: {benchmark.json_validity} валидных JSON из {len(TESTS)}")
        print(f"{model_name}: {benchmark.valid_objects} валидных объектов из {TOTAL_OBJECTS}")
        print(f"{model_name}: {benchmark.valid_scenes} правильных сцен из {len(TESTS)}")


if __name__ == "__main__":
    main()
