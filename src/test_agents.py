from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore, extract_profile_updates, summarize_messages

ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path, **overrides):
    """Isolated offline config: state lives in tmp_path, small compact threshold."""

    values = {
        "state_dir": tmp_path / "state",
        "compact_threshold_tokens": 60,
        "compact_keep_messages": 2,
        "live": False,
    }
    values.update(overrides)
    return dataclasses.replace(load_config(ROOT), **values)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    user = "dungct"

    assert store.file_size(user) == 0
    assert "## Facts" in store.read_text(user)  # default template before the file exists

    path = store.write_text(user, "# User Profile: dungct\n\n## Facts\n\n- name: DũngCT\n")
    assert path.exists() and path.name == "User.md"
    assert store.facts(user) == {"name": "DũngCT"}

    assert store.edit_text(user, "DũngCT", "DũngCT Stress") is True
    assert store.facts(user)["name"] == "DũngCT Stress"
    assert store.edit_text(user, "not-in-file", "x") is False

    assert store.upsert_fact(user, "location", "Đà Nẵng") is True
    assert store.upsert_fact(user, "location", "Đà Nẵng") is False  # no-op write
    assert store.upsert_fact(user, "location", "Huế") is True
    assert store.facts(user)["location"] == "Huế"
    assert "location: Đà Nẵng -> Huế" in store.read_text(user)
    assert store.file_size(user) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    long_turn = "Mình kể thêm một đoạn dài về pipeline MLOps và chi phí ngữ cảnh. " * 5

    for _ in range(4):
        agent.reply("u1", "long-thread", long_turn)

    ctx = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") >= 1
    assert len(ctx["messages"]) <= agent.config.compact_keep_messages + 1
    assert ctx["summary"]
    # a short thread stays below the threshold and is never compacted
    agent.reply("u1", "short-thread", "Chào bạn.")
    assert agent.compaction_count("short-thread") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    turns = ["Chào bạn, mình tên là DũngCT.", "Đồ uống yêu thích là cà phê sữa đá."]
    question = "Mình tên gì và đồ uống yêu thích là gì?"

    for turn in turns:
        advanced.reply("dungct", "session-1", turn)
        baseline.reply("dungct", "session-1", turn)

    # baseline remembers inside the same thread...
    assert "DũngCT" in baseline.reply("dungct", "session-1", question)["response"]
    # ...but forgets in a new thread, while advanced recalls from User.md
    baseline_answer = baseline.reply("dungct", "session-2", question)["response"]
    advanced_answer = advanced.reply("dungct", "session-2", question)["response"]
    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer and "cà phê sữa đá" in advanced_answer

    # persistence survives a brand-new agent instance (process restart)
    restarted = AdvancedAgent(config, force_offline=True)
    assert "DũngCT" in restarted.reply("dungct", "session-3", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, compact_threshold_tokens=500, compact_keep_messages=4)
    conversation = json.loads((ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))[0]
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    for turn in conversation["turns"]:
        advanced.reply(conversation["user_id"], "stress", turn)
        baseline.reply(conversation["user_id"], "stress", turn)

    assert advanced.compaction_count("stress") > 1
    assert advanced.prompt_token_usage("stress") < 0.7 * baseline.prompt_token_usage("stress")


def test_summary_keeps_informative_content_and_drops_acks() -> None:
    summary = summarize_messages(
        [
            {"role": "user", "content": "Mình kể thêm một chút. Tin thứ hai là bài NASA năm 2026 về X-59."},
            {"role": "assistant", "content": "Đã ghi nhận. Cập nhật User.md: Tên = DũngCT."},
            {"role": "assistant", "content": "- Tên: DũngCT"},
        ]
    )
    assert "X-59" in summary and "Mình kể thêm" not in summary
    assert "Đã ghi nhận" not in summary and "- Tên" not in summary


def test_correction_keeps_latest_fact_only(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("u", "t1", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    agent.reply("u", "t1", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    agent.reply("u", "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    facts = agent.profile_store.facts("u")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    answer = agent.reply("u", "t2", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "backend" not in answer and "Đà Nẵng" not in answer


def test_noise_questions_and_hedges_are_not_saved() -> None:
    joke = "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa."
    trip = "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày chứ không phải nơi ở hiện tại."
    assert "profession" not in extract_profile_updates(joke)
    assert "location" not in extract_profile_updates(trip)
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Mình tên gì?") == {}
    # confidence threshold: hedged statements stay out of User.md
    assert "location" not in extract_profile_updates("Có lẽ mình sẽ chuyển ra Hà Nội.")
    assert extract_profile_updates("Mình chuyển ra Hà Nội rồi.")["location"] == "Hà Nội"
