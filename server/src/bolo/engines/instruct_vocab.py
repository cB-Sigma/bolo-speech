"""Plain-language ``instructions`` turned into the fixed items a model understands.

Some models take a direction in free words (Qwen3-TTS CustomVoice, CosyVoice3, Maya1, Parler-TTS). OmniVoice takes
only a closed list of items, comma-separated ("female, low pitch, british accent", "whisper"), and audio.cpp refuses
anything else. :func:`omnivoice` finds those items in ordinary words ("Whisper very softly", "an old man with a deep
voice", "speak with an Indian accent") and says which words it could not place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: OmniVoice's items (audio.cpp ``src/models/omnivoice/prompt_builder.cpp``), by category: one item per category
OMNIVOICE_ITEMS: dict[str, tuple[str, ...]] = {
    "gender": ("male", "female"),
    "age": ("child", "teenager", "young adult", "middle-aged", "elderly"),
    "pitch": ("very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"),
    "style": ("whisper",),
    "accent": (
        "american accent", "british accent", "australian accent", "chinese accent", "canadian accent",
        "indian accent", "korean accent", "portuguese accent", "russian accent", "japanese accent",
    ),
}  # fmt: skip
OMNIVOICE_DIALECTS = (
    "河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话",
)  # fmt: skip
_OMNIVOICE_ZH = {
    "男": "male", "女": "female", "儿童": "child", "少年": "teenager", "青年": "young adult", "中年": "middle-aged",
    "老年": "elderly", "极低音调": "very low pitch", "低音调": "low pitch", "中音调": "moderate pitch",
    "高音调": "high pitch", "极高音调": "very high pitch", "耳语": "whisper",
}  # fmt: skip

# plain words -> item, tried in order (longer phrases first); each pattern is matched on whole words
_RULES: list[tuple[str, str, str]] = [
    # pitch
    ("pitch", r"very (?:low|deep)(?:[- ]pitch(?:ed)?)?|extremely (?:low|deep)|very deep voice", "very low pitch"),
    ("pitch", r"very high(?:[- ]pitch(?:ed)?)?|extremely high|squeaky", "very high pitch"),
    ("pitch", r"low[- ]pitch(?:ed)?|deep(?:er)? voice|deep|low voice|bass|baritone", "low pitch"),
    ("pitch", r"high[- ]pitch(?:ed)?|high voice|shrill|falsetto", "high pitch"),
    ("pitch", r"(?:moderate|medium|normal|average)[- ]pitch(?:ed)?", "moderate pitch"),
    # style
    ("style", r"whisper(?:s|ed|ing|y)?|softly|quietly|hushed|under (?:his|her|their|one's) breath", "whisper"),
    # gender (girl / boy also say an age)
    ("gender", r"female|woman|women|lady|feminine|girl|she", "female"),
    ("gender", r"male|man|men|gentleman|masculine|boy|he", "male"),
    # age
    ("age", r"child|kid|little (?:girl|boy)|young (?:girl|boy)|girl|boy|toddler", "child"),
    ("age", r"teen(?:ager|age|aged)?|adolescent", "teenager"),
    ("age", r"middle[- ]aged|middle age", "middle-aged"),
    ("age", r"elderly|old(?:er)?|aged|senior|grandm(?:a|other)|grandp(?:a|father)", "elderly"),
    ("age", r"young adult|young|youthful", "young adult"),
    # accent
    ("accent", r"american", "american accent"),
    ("accent", r"british|uk|u\.k\.|london", "british accent"),
    ("accent", r"australian|aussie", "australian accent"),
    ("accent", r"chinese|mandarin", "chinese accent"),
    ("accent", r"canadian", "canadian accent"),
    ("accent", r"indian", "indian accent"),
    ("accent", r"korean", "korean accent"),
    ("accent", r"portuguese|brazilian", "portuguese accent"),
    ("accent", r"russian", "russian accent"),
    ("accent", r"japanese", "japanese accent"),
]
# words that carry no direction of their own ("speak in a whisper with a british accent")
_FILLER = {
    "a", "an", "the", "in", "with", "and", "of", "to", "for", "as", "at", "on", "very", "really", "quite",
    "rather", "slightly", "bit", "little", "somewhat", "please", "speak", "speaks", "speaking", "say", "says",
    "said", "talk", "talking", "read", "reading", "sound", "sounds", "sounding", "like", "voice", "voiced", "tone",
    "accent", "accented", "pitch", "pitched", "person", "speaker", "narrator", "someone", "who", "is", "are", "be",
    "that", "this", "it", "but", "or", "so", "tell",
}  # fmt: skip


@dataclass
class Mapped:
    """What became of plain instructions: the model's items, and the words that matched none."""

    items: list[str] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return ", ".join(self.items)


def omnivoice(instructions: str) -> Mapped:
    """OmniVoice's items for ``instructions``: exact items pass through; plain words are matched to items (one per
    category, the first that matches); words that place nowhere come back in ``unplaced``."""
    out = Mapped()
    taken: set[str] = set()
    text = instructions.strip()
    exact = [p.strip() for p in re.split(r"\s*[,，]\s*", text) if p.strip()]
    valid = {i for items in OMNIVOICE_ITEMS.values() for i in items}
    if exact and all(p.lower() in valid or p in OMNIVOICE_DIALECTS or p in _OMNIVOICE_ZH for p in exact):
        for p in exact:
            item = _OMNIVOICE_ZH.get(p, p if p in OMNIVOICE_DIALECTS else p.lower())
            category = next((c for c, items in OMNIVOICE_ITEMS.items() if item in items), "dialect")
            if category not in taken:
                out.items.append(item)
                taken.add(category)
        return out
    low = f" {text.lower()} "
    rest = low
    for category, pattern, item in _RULES:
        regex = re.compile(rf"(?<![\w-])(?:{pattern})(?![\w-])")
        if regex.search(low) is None:
            continue
        rest = regex.sub(" ", rest)
        if category not in taken:
            out.items.append(item)
            taken.add(category)
    for dialect in OMNIVOICE_DIALECTS:
        if dialect in rest and "dialect" not in taken and "accent" not in taken:
            out.items.append(dialect)
            taken.add("dialect")
            rest = rest.replace(dialect, " ")
    out.unplaced = [w for w in re.findall(r"[^\W\d_][\w'-]*", rest) if w not in _FILLER]
    return out


def omnivoice_items_list() -> str:
    """The items OmniVoice understands, for messages."""
    return "; ".join(", ".join(items) for items in OMNIVOICE_ITEMS.values())
