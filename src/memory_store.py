from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate: ~4 characters per token, 0 for empty text."""

    stripped = (text or "").strip()
    if not stripped:
        return 0
    return math.ceil(len(stripped) / 4)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

FACTS_HEADER = "## Facts"
CHANGELOG_HEADER = "## Changelog"
MAX_CHANGELOG_ENTRIES = 20
_FACT_LINE = re.compile(r"^- ([a-z_]+): (.+)$")


def _change_entry(key: str, old: str, new: str) -> str | None:
    """Scalar facts log `old -> new`; list facts log only the diff (+added / -removed)."""

    if key not in LIST_FACTS:
        return f"{key}: {old} -> {new}"
    old_items = {i.strip() for i in old.split(";")}
    new_items = [i.strip() for i in new.split(";")]
    added = [f"+{i}" for i in new_items if i not in old_items]
    removed = [f"-{i}" for i in old_items - set(new_items)]
    return f"{key}: {', '.join(added + sorted(removed))}" if added or removed else None


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`: one markdown file per user.

    Layout:
        # User Profile: <user_id>
        ## Facts          -> "- key: value" lines, always the latest value
        ## Changelog      -> "- key: old -> new" lines, capped to avoid unbounded growth
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id.strip()).strip("_") or "anonymous"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return self._render(user_id, {}, [])

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists() or not search_text:
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts, _ = self._parse(self.read_text(user_id))
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Insert or replace one fact. Corrections are logged in the changelog.

        Returns True when the file changed.
        """

        value = " ".join(value.split())
        facts, changelog = self._parse(self.read_text(user_id))
        old = facts.get(key)
        if old == value:
            return False
        if old is not None:
            entry = _change_entry(key, old, value)
            if entry:
                changelog = (changelog + [entry])[-MAX_CHANGELOG_ENTRIES:]
        facts[key] = value
        self.write_text(user_id, self._render(user_id, facts, changelog))
        return True

    @staticmethod
    def _parse(content: str) -> tuple[dict[str, str], list[str]]:
        facts: dict[str, str] = {}
        changelog: list[str] = []
        section = None
        for raw in content.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                section = line
                continue
            if section == FACTS_HEADER:
                match = _FACT_LINE.match(line)
                if match:
                    facts[match.group(1)] = match.group(2)
            elif section == CHANGELOG_HEADER and line.startswith("- "):
                changelog.append(line[2:])
        return facts, changelog

    @staticmethod
    def _render(user_id: str, facts: dict[str, str], changelog: list[str]) -> str:
        lines = [f"# User Profile: {user_id}", "", FACTS_HEADER, ""]
        lines += [f"- {key}: {value}" for key, value in facts.items()]
        lines += ["", CHANGELOG_HEADER, ""]
        lines += [f"- {entry}" for entry in changelog]
        return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Fact extraction (heuristic, Vietnamese)
# ---------------------------------------------------------------------------

CITIES = (
    "Đà Nẵng", "Huế", "Hà Nội", "Hồ Chí Minh", "TP.HCM", "Sài Gòn", "Hải Phòng",
    "Cần Thơ", "Nha Trang", "Đà Lạt", "Quy Nhơn", "Vũng Tàu", "Hội An",
)
_CITY = "(" + "|".join(re.escape(c) for c in CITIES) + ")"
_LOCATION_PATTERNS = [
    re.compile(r"từ\s+" + _CITY + r"\s+sang\s+" + _CITY),  # "cập nhật từ Huế sang Đà Nẵng" -> last group
    re.compile(r"nơi ở(?:\s+hiện tại)?\s+(?:là|ở)\s+" + _CITY, re.IGNORECASE),
    re.compile(r"(?:chuyển|dọn)\s+(?:ra|về|vào|đến|tới|sang)\s+" + _CITY, re.IGNORECASE),
    re.compile(r"(?:sống|ở|làm việc)\s+(?:ở|tại)\s+" + _CITY, re.IGNORECASE),
    re.compile(r"\bở\s+" + _CITY, re.IGNORECASE),
]

_ROLE = r"((?:[A-Za-z][A-Za-z0-9+-]*\s+){0,2}?(?:engineer|developer|manager|scientist|analyst|designer|researcher|architect))\b"
_PROFESSION_PATTERN = re.compile(r"(?:làm|là|sang|nghề)\s+" + _ROLE, re.IGNORECASE)

_NAME_PATTERNS = [
    re.compile(r"(?:mình|tôi|em)\s+tên\s+là\s+(.+)", re.IGNORECASE),
    re.compile(r"tên\s+(?:của\s+)?(?:mình|tôi|em)\s+là\s+(.+)", re.IGNORECASE),
    re.compile(r"^\s*tên\s+(?:là\s+)?(.+)", re.IGNORECASE),  # clause starting with "tên ..."
]

_DRINK_PATTERN = re.compile(r"đồ uống\s+(?:yêu thích|ưa thích)(?:\s+của\s+(?:mình|tôi))?\s+là\s+([^.,;!?]+)", re.IGNORECASE)
_FOOD_PATTERN = re.compile(r"món(?:\s+ăn)?\s+(?:yêu thích|ưa thích)(?:\s+của\s+(?:mình|tôi))?\s+là\s+([^.,;!?]+)", re.IGNORECASE)
_PET_PATTERN = re.compile(r"nuôi\s+(?:một\s+)?(?:bé\s+|con\s+|chú\s+|em\s+)?([\w-]+)\s+tên\s+(\w+)", re.IGNORECASE)

_STYLE_FEATURES = [
    (re.compile(r"ngắn gọn|trả lời ngắn|bullet ngắn|\bgọn\b", re.IGNORECASE), "ngắn gọn"),
    (re.compile(r"rõ ý", re.IGNORECASE), "rõ ý"),
    (re.compile(r"có cấu trúc", re.IGNORECASE), "có cấu trúc"),
    (re.compile(r"ví dụ thực tế", re.IGNORECASE), "có ví dụ thực tế"),
    (re.compile(r"ví dụ thực chiến", re.IGNORECASE), "có ví dụ thực chiến"),
    (re.compile(r"trade-off", re.IGNORECASE), "nhấn trade-off"),
    (re.compile(r"định lượng|số liệu", re.IGNORECASE), "có số liệu minh họa"),
]
_BULLET_COUNT = re.compile(r"(\d+)\s*bullet", re.IGNORECASE)
_BULLET_ANY = re.compile(r"\bbullet\b", re.IGNORECASE)

TECH_TERMS = (
    "AI ứng dụng", "AI agent", "async Python", "memory architecture", "benchmark memory",
    "LangGraph", "LangChain", "MLOps", "Python", "RAG", "evaluation",
)

_QUESTION_WORDS = {"gì", "ai", "đâu", "nào", "sao"}
_NEGATION = re.compile(r"\b(?:không|chưa|đừng|chỉ là|lúc đầu|trước đó|trước đây|đã từng)\b", re.IGNORECASE)
_HYPOTHETICAL = re.compile(r"\b(?:đùa|giả sử|giả dụ)\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:có lẽ|hình như|chắc là|có thể sẽ|dự định|đang định|đang cân nhắc|chưa chắc)\b", re.IGNORECASE)
_CORRECTION = re.compile(r"\b(?:đính chính|thực ra|giờ|hiện tại|hiện|cập nhật|vẫn|nhớ)\b", re.IGNORECASE)
_PREFERENCE_TOPIC = re.compile(r"trả lời|giải thích|style", re.IGNORECASE)
_PREFERENCE_VERB = re.compile(r"\b(?:muốn|thích|hãy|giữ|ưu tiên|nhớ)\b", re.IGNORECASE)
_INTEREST_VERB = re.compile(r"\b(?:thích|quan tâm|đang học|học thêm)\b", re.IGNORECASE)

BASE_CONFIDENCE = 0.7


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _split_clauses(sentence: str) -> list[str]:
    parts = re.split(r"[,;:]|\s+(?:chứ|nhưng|dù|song)\s+", sentence)
    return [p.strip() for p in parts if p and p.strip()]


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().endswith("?")


def _clean_value(value: str) -> str | None:
    value = re.sub(r"\s+(?:nhé|nha|ạ|đó|như cũ)$", "", value.strip(), flags=re.IGNORECASE).strip()
    words = value.lower().split()
    if not words or any(w in _QUESTION_WORDS for w in words):
        return None
    return value


def _take_name(rest: str) -> str | None:
    """Keep leading capitalised words: 'DũngCT Stress, hiện ở Huế' -> 'DũngCT Stress'."""

    words: list[str] = []
    for word in rest.split():
        bare = word.strip(".,;:!?\"'()")
        if not bare or not bare[0].isupper():
            break
        words.append(bare)
        if bare != word:  # punctuation ended the name
            break
    return " ".join(words) or None


def _confidence(sentence: str) -> float:
    score = BASE_CONFIDENCE
    if _CORRECTION.search(sentence):
        score += 0.2
    if _HEDGE.search(sentence):
        score -= 0.5
    return round(max(0.0, min(1.0, score)), 2)


def extract_profile_candidates(message: str) -> dict[str, tuple[str, float]]:
    """Extract (value, confidence) for each stable fact found in the message.

    Guards against the traps in the dataset:
    - question sentences never write facts ("Mình tên gì?")
    - negated clauses are ignored ("không còn ở Đà Nẵng", "chỉ là nơi đi họp")
    - joke / hypothetical sentences are ignored ("đùa ... product manager")
    - when a message mentions several values, the last valid one wins (corrections)
    - list-like facts (style, interests) are returned as "; "-joined features
    """

    found: dict[str, tuple[str, float]] = {}
    style: list[str] = []
    interests: list[str] = []
    style_conf = interest_conf = 0.0

    for sentence in _split_sentences(_nfc(message)):
        if _is_question(sentence):
            continue
        conf = _confidence(sentence)
        identity_ok = not _HYPOTHETICAL.search(sentence)

        if identity_ok:
            for clause in _split_clauses(sentence):
                negated = bool(_NEGATION.search(clause))

                for pattern in _NAME_PATTERNS:
                    match = pattern.search(clause)
                    if match and not negated:
                        name = _take_name(match.group(1))
                        if name:
                            found["name"] = (name, conf)
                        break

                if not negated:
                    for pattern in _LOCATION_PATTERNS:
                        match = pattern.search(clause)
                        if match:
                            found["location"] = (match.group(match.lastindex), conf)
                            break

                    match = _PROFESSION_PATTERN.search(clause)
                    if match:
                        found["profession"] = (" ".join(match.group(1).split()), conf)

        for key, pattern in (("favorite_drink", _DRINK_PATTERN), ("favorite_food", _FOOD_PATTERN)):
            match = pattern.search(sentence)
            if match:
                value = _clean_value(match.group(1))
                if value:
                    found[key] = (value, conf)

        match = _PET_PATTERN.search(sentence)
        if match and match.group(2)[0].isupper():
            found["pet"] = (f"{match.group(1)} tên {match.group(2)}", conf)

        if _PREFERENCE_TOPIC.search(sentence) and _PREFERENCE_VERB.search(sentence):
            for pattern, label in _STYLE_FEATURES:
                if pattern.search(sentence) and label not in style:
                    style.append(label)
            count = _BULLET_COUNT.search(sentence)
            if count:
                style.append(f"{count.group(1)} bullet")
            elif _BULLET_ANY.search(sentence):
                style.append("dạng bullet")
            if style:
                style_conf = max(style_conf, conf)

        if _INTEREST_VERB.search(sentence):
            remaining = sentence
            for term in TECH_TERMS:  # longest/specific terms first so "async Python" wins over "Python"
                if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", remaining):
                    interests.append(term)
                    remaining = remaining.replace(term, " ")
            if interests:
                interest_conf = max(interest_conf, conf)

    if style:
        found["style"] = ("; ".join(dict.fromkeys(style)), style_conf)
    if interests:
        found["interests"] = ("; ".join(dict.fromkeys(interests)), interest_conf)
    return found


def extract_profile_updates(message: str, min_confidence: float = 0.5) -> dict[str, str]:
    """Convert raw user text into stable profile facts, keeping only confident ones."""

    return {
        key: value
        for key, (value, confidence) in extract_profile_candidates(message).items()
        if confidence >= min_confidence
    }


LIST_FACTS = {"style": 8, "interests": 6}


def merge_fact(key: str, old: str | None, new: str) -> str:
    """Scalar facts are replaced (latest correction wins); list facts are merged.

    For `style`, a new "N bullet" replaces the old count and generic "dạng bullet".
    List facts are capped (newest kept) so User.md cannot grow without bound.
    """

    if key not in LIST_FACTS or not old:
        return new
    old_items = [item.strip() for item in old.split(";") if item.strip()]
    new_items = [item.strip() for item in new.split(";") if item.strip()]
    if key == "style" and any(_BULLET_COUNT.fullmatch(item) for item in new_items):
        old_items = [i for i in old_items if not _BULLET_COUNT.fullmatch(i) and i != "dạng bullet"]
    if key == "style" and any(_BULLET_COUNT.fullmatch(item) for item in old_items):
        new_items = [i for i in new_items if i != "dạng bullet"]
    merged = list(dict.fromkeys(old_items + new_items))
    return "; ".join(merged[-LIST_FACTS[key]:])


# ---------------------------------------------------------------------------
# Recall answers shared by both agents (offline mode)
# ---------------------------------------------------------------------------

FIELD_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "style": "Style trả lời",
    "interests": "Mối quan tâm chính",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}
_FIELD_CUES = {
    "name": ("tên", "là ai"),
    "location": ("ở đâu", "nơi ở", "còn ở", "đang ở", "sống ở"),
    "profession": ("nghề", "là ai", "làm gì"),
    "style": ("style", "kiểu trả lời", "trả lời như thế nào", "cách trả lời"),
    "interests": ("quan tâm", "là ai"),
    "favorite_drink": ("đồ uống", "uống gì"),
    "favorite_food": ("món ăn", "ăn gì"),
    "pet": ("nuôi", "thú cưng"),
}


def requested_fields(question: str) -> list[str]:
    text = _nfc(question).lower()
    return [key for key, cues in _FIELD_CUES.items() if any(cue in text for cue in cues)]


def render_recall_answer(question: str, facts: dict[str, str]) -> str | None:
    """Answer a recall question from known facts as short bullets; None if nothing is asked."""

    fields = requested_fields(question)
    if not fields:
        return None
    lines = [f"- {FIELD_LABELS[key]}: {facts.get(key, 'mình chưa có thông tin')}" for key in fields]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


_LOW_INFO = re.compile(r"^đã ghi nhận", re.IGNORECASE)  # acks (+ "Cập nhật User.md" echoes)


def _information_score(sentence: str) -> int:
    """Proper nouns / acronyms (not the first word) and numbers carry most of the content."""

    words = sentence.split()
    named = sum(1 for w in words[1:] if w[:1].isupper() or any(ch.isdigit() for ch in w))
    return named


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages, one line each.

    - Acknowledgements ("Đã ghi nhận. Cập nhật User.md: ...") are dropped: they carry nothing new.
    - Assistant recall answers are dropped too: they only echo User.md, which is already in the prompt.
    - For user messages, keep the most informative sentence (names, numbers) rather than the
      first one, which in long turns is often a filler opener.
    """

    lines = []
    for message in messages[-max_items:]:
        content = " ".join(message.get("content", "").split())
        role = message.get("role", "user")
        if not content or _LOW_INFO.match(content) or (role == "assistant" and content.startswith("- ")):
            continue
        sentences = _split_sentences(content)
        best = max(sentences, key=lambda s: (_information_score(s), -sentences.index(s)))
        if len(best) > 140:
            best = best[:137].rstrip() + "..."
        lines.append(f"- {role}: {best}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Recent messages are kept verbatim.
    - When the kept messages exceed `threshold_tokens`, everything except the last
      `keep_messages` is folded into a bounded summary.
    - `compactions` counts how often that happened (benchmark column).
    """

    threshold_tokens: int
    keep_messages: int
    max_summary_lines: int = 10
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        ctx = self.context(thread_id)
        ctx["messages"].append({"role": role, "content": content})
        if self.message_tokens(thread_id) > self.threshold_tokens:
            self._compact(ctx)

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])

    def message_tokens(self, thread_id: str) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.context(thread_id)["messages"])

    def _compact(self, ctx: dict[str, object]) -> None:
        messages: list[dict[str, str]] = ctx["messages"]
        keep = max(self.keep_messages, 0)
        older, recent = (messages[:-keep], messages[-keep:]) if keep else (messages, [])
        if not older:
            return
        summary_lines = [l for l in str(ctx["summary"]).splitlines() if l.strip()]
        summary_lines += summarize_messages(older, max_items=len(older)).splitlines()
        ctx["summary"] = "\n".join(summary_lines[-self.max_summary_lines:])
        ctx["messages"] = recent
        ctx["compactions"] = int(ctx["compactions"]) + 1
