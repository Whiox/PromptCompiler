"""Run with: python -m streamlit run app.py"""

import json
from pathlib import Path
from threading import Lock
from time import perf_counter

import streamlit as st

from inference import Model


ROOT = Path(__file__).resolve().parent
EXAMPLES = [
    ("Чашка и коробка", "Show the red ceramic cup next to the blue metal box."),
    ("На столе", "The green glass bottle is on the brown wooden table."),
    ("Слева", "Place the red cup to the left of the blue box."),
    ("Без материала", "Show the red cup next to the blue box."),
    ("Другая формулировка", "The cup is ceramic and green. It is under the wooden table."),
]


@st.cache_resource(max_entries=1, show_spinner="Загружаю модель…")
def load_model(name: str, checkpoint_modified: int):
    # mtime is part of the cache key so retraining refreshes the cached model.
    return Model(experiment_name=name), Lock()


def compile_scene(scene: dict) -> str:
    """Validate supported fields and render the model's scene without enrichment."""
    if not isinstance(scene, dict):
        raise ValueError("Ожидался JSON-объект со сценой.")
    objects, relations = scene.get("objects"), scene.get("relations")
    if not isinstance(objects, list) or not objects or not isinstance(relations, list):
        raise ValueError("Нужны непустой список objects и список relations.")

    descriptions = {}
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Каждый объект должен быть словарём.")
        object_id, name = obj.get("id"), obj.get("name")
        if not isinstance(object_id, str) or not object_id or object_id in descriptions:
            raise ValueError("У объектов должны быть уникальные строковые id.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("У объекта отсутствует название.")
        color, material = obj.get("color"), obj.get("material")
        if any(value is not None and not isinstance(value, str) for value in (color, material)):
            raise ValueError("Цвет и материал должны быть строкой или null.")
        material = "wooden" if material == "wood" else material
        descriptions[object_id] = " ".join(value for value in (color, material, name) if value)

    parts = ["Objects: " + "; ".join(descriptions.values()) + "."]
    supported_relations = {"next to", "under", "on", "above", "behind", "in front of", "to the left of", "to the right of"}
    for relation in relations:
        if not isinstance(relation, dict):
            raise ValueError("Некорректное отношение между объектами.")
        subject, other, kind = relation.get("subject"), relation.get("object"), relation.get("type")
        if not all(isinstance(value, str) for value in (subject, other, kind)):
            raise ValueError("Поля отношения должны быть строками.")
        if subject not in descriptions or other not in descriptions or kind not in supported_relations:
            raise ValueError("Неизвестное отношение или ссылка на отсутствующий объект.")
        parts.append(f"The {descriptions[subject]} is {kind} the {descriptions[other]}.")

    for field in ("background", "style", "lighting"):
        value = scene.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Поле {field} должно быть строкой или null.")
        if value:
            parts.append(f"{field.capitalize()}: {value}.")
    composition = scene.get("composition", {})
    if not isinstance(composition, dict):
        raise ValueError("Поле composition должно быть объектом.")
    for field, label in (("subject_position", "Subject position"), ("negative_space", "Negative space")):
        value = composition.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Поле composition.{field} должно быть строкой или null.")
        if value:
            parts.append(f"{label}: {value}.")
    forbidden = scene.get("forbidden", [])
    if not isinstance(forbidden, list) or any(not isinstance(value, str) for value in forbidden):
        raise ValueError("Поле forbidden должно быть списком строк.")
    if forbidden:
        parts.append("Do not include: " + ", ".join(forbidden) + ".")
    return " ".join(parts)


def show_answer(message: dict, index: int) -> None:
    st.caption(f"{message['model']} · {message['seconds']:.2f} с")
    if message.get("error"):
        st.error(message["error"])
    else:
        st.markdown("**Готовый промпт**")
        st.code(message["prompt"], language=None, wrap_lines=True)
        st.markdown("**Объекты сцены**")
        st.dataframe(message["scene"]["objects"], hide_index=True, width="stretch")
        st.caption("Проверь, что объекты и свойства совпадают с твоим описанием.")
    with st.expander("Ответ модели · JSON"):
        if message.get("scene") is not None:
            st.json(message["scene"])
        else:
            st.code(message.get("raw", ""), language="json")
    if message.get("prompt"):
        st.download_button("Скачать JSON", json.dumps(message["scene"], ensure_ascii=False, indent=2), file_name="scene.json", mime="application/json", key=f"download_{index}")


def main() -> None:
    st.set_page_config(page_title="PromptCompiler", page_icon="✦", layout="centered")
    st.title("PromptCompiler")
    st.markdown("Преврати описание изображения в структуру сцены и готовый промпт.")
    st.caption("Пиши по-английски. Каждое сообщение — самостоятельная сцена с двумя объектами; уточнения предыдущего сообщения пока не поддерживаются.")

    checkpoints = sorted((ROOT / "experiments").glob("*/ckpt.pt"))
    if not checkpoints:
        st.info("Добавь обученную модель: experiments/<имя>/ckpt.pt")
        st.stop()
    names = [path.parent.name for path in checkpoints]
    preferred = "nanogpt_v0.4_2_seed42"
    example_request = None
    with st.sidebar:
        st.header("Настройки")
        selected = st.selectbox("Модель", names, index=names.index(preferred) if preferred in names else len(names) - 1)
        st.caption("GPT-2 · nanoGPT · локальный запуск")
        if st.button("Очистить историю", width="stretch"):
            st.session_state.messages = []
        st.divider()
        st.subheader("Примеры запросов")
        for label, request in EXAMPLES:
            if st.button(label, key=f"example_{label}", width="stretch"):
                example_request = request

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                show_answer(message, index)

    request = st.chat_input("Describe your scene…", max_chars=1500)
    request = example_request or request
    if not request or not request.strip():
        return
    st.session_state.messages.append({"role": "user", "content": request})
    with st.chat_message("user"):
        st.write(request)
    answer = {"role": "assistant", "model": selected, "raw": "", "seconds": 0.0}
    started = perf_counter()
    with st.chat_message("assistant"):
        try:
            checkpoint = ROOT / "experiments" / selected / "ckpt.pt"
            model, lock = load_model(selected, checkpoint.stat().st_mtime_ns)
            with st.spinner("Разбираю сцену…"), lock:
                answer["raw"] = model.generate_scene(request)
            try:
                answer["scene"] = json.loads(answer["raw"])
            except json.JSONDecodeError:
                raise ValueError("Модель вернула некорректный JSON. Попробуй упростить описание или выбрать другую модель.") from None
            answer["prompt"] = compile_scene(answer["scene"])
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            answer["error"] = str(error)
        answer["seconds"] = perf_counter() - started
        st.session_state.messages.append(answer)
        show_answer(answer, len(st.session_state.messages) - 1)


if __name__ == "__main__":
    main()
