"""Named-entity recognition for research text (§46).

Honest availability: spaCy + a loaded model are required. This module never
fabricates entities when the stack is missing.
"""

from __future__ import annotations

from typing import Any


def ner_available(model_name: str = "en_core_web_sm") -> bool:
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    try:
        import spacy

        spacy.load(model_name)
        return True
    except Exception:  # noqa: BLE001 — any load failure means unavailable
        return False


def extract_entities(
    texts: list[str],
    *,
    model_name: str = "en_core_web_sm",
    unit_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run spaCy NER when available; otherwise raise a clear ValueError."""
    if unit_ids is not None and len(unit_ids) != len(texts):
        raise ValueError("unit_ids must align 1:1 with texts")
    try:
        import spacy
    except ImportError as exc:
        raise ValueError(
            "NER requires the optional spaCy package; it is not a core dependency."
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"NER model {model_name!r} is not installed or failed to load: {exc}. "
            "Install the model explicitly if NER is required for a project."
        ) from exc

    entities: list[dict[str, Any]] = []
    for i, text in enumerate(texts):
        doc = nlp(text or "")
        unit_id = unit_ids[i] if unit_ids else str(i)
        for ent in doc.ents:
            entities.append(
                {
                    "text_unit_id": unit_id,
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
    return {
        "available": True,
        "model": model_name,
        "n_entities": len(entities),
        "entities": entities,
    }


def describe_ner_capabilities() -> dict[str, Any]:
    return {
        "available": ner_available(),
        "default_model": "en_core_web_sm",
        "required": False,
        "notes": ["NER is optional; analyses degrade honestly when spaCy/model missing."],
    }
