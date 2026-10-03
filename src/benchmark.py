from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import estimate_tokens

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _coverage(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    text = _norm(answer)
    return sum(1 for item in expected if _norm(item) in text) / len(expected)


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    coverage = _coverage(answer, expected)
    if coverage == 1.0:
        return 1.0
    return 0.5 if coverage > 0 else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    - 0.70 * fact coverage (correct content matters most)
    - 0.15 if concise (<= 80 tokens), matching the user's "ngắn gọn" preference
    - 0.15 if structured as bullets
    - minus 0.10 when the agent admits it has no information
    """

    score = 0.7 * _coverage(answer, expected)
    if estimate_tokens(answer) <= 80:
        score += 0.15
    if any(line.lstrip().startswith(("-", "*", "•")) for line in answer.splitlines()):
        score += 0.15
    if "chưa có thông tin" in _norm(answer):
        score -= 0.1
    return round(max(0.0, min(1.0, score)), 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Feed every conversation, then ask the recall questions in a fresh thread."""

    user_ids = sorted({conv["user_id"] for conv in conversations})
    has_memory_file = hasattr(agent, "memory_file_size")
    size_before = sum(agent.memory_file_size(u) for u in user_ids) if has_memory_file else 0

    threads: list[str] = []
    compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        thread_id = conv["id"]
        threads.append(thread_id)
        for turn in conv["turns"]:
            agent.reply(conv["user_id"], thread_id, turn)
        compactions += agent.compaction_count(thread_id)

        for index, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{thread_id}::recall-{index}"  # new thread = new session
            threads.append(recall_thread)
            answer = agent.reply(conv["user_id"], recall_thread, item["question"])["response"]
            recall_scores.append(recall_points(answer, item["expected_contains"]))
            quality_scores.append(heuristic_quality(answer, item["expected_contains"]))

    size_after = sum(agent.memory_file_size(u) for u in user_ids) if has_memory_file else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(agent.token_usage(t) for t in threads),
        prompt_tokens_processed=sum(agent.prompt_token_usage(t) for t in threads),
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=size_after - size_before,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Markdown table with the six required columns."""

    table = [
        [
            row.agent_name,
            f"{row.agent_tokens_only:,}",
            f"{row.prompt_tokens_processed:,}",
            f"{row.recall_score:.0%}",
            f"{row.response_quality:.2f}",
            f"{row.memory_growth_bytes:,}",
            str(row.compactions),
        ]
        for row in rows
    ]
    widths = [max(len(COLUMNS[i]), *(len(r[i]) for r in table)) for i in range(len(COLUMNS))]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(COLUMNS, widths)) + " |"
    divider = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in table]
    return "\n".join([header, divider, *body])


def _compare(baseline: BenchmarkRow, advanced: BenchmarkRow) -> str:
    def delta(new: int, old: int) -> str:
        return f"{(new - old) / old:+.0%}" if old else "n/a"

    return (
        f"Advanced vs Baseline: agent tokens {delta(advanced.agent_tokens_only, baseline.agent_tokens_only)}, "
        f"prompt tokens {delta(advanced.prompt_tokens_processed, baseline.prompt_tokens_processed)}, "
        f"recall {advanced.recall_score - baseline.recall_score:+.0%}"
    )


def _fresh_config(config: LabConfig, state_name: str, **overrides) -> LabConfig:
    state_dir = config.state_dir / "benchmark" / state_name
    shutil.rmtree(state_dir, ignore_errors=True)  # every run starts from an empty User.md
    return dataclasses.replace(config, state_dir=state_dir, **overrides)


def run_suite(title: str, dataset: Path, config: LabConfig, state_name: str) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    force_offline = not config.live
    suite_config = _fresh_config(config, state_name)
    # Ablation: same agent with compaction disabled, to isolate what compact (vs User.md) contributes.
    no_compact_config = _fresh_config(config, f"{state_name}-no-compact", compact_threshold_tokens=10**9)

    rows = [
        run_agent_benchmark("Baseline", BaselineAgent(suite_config, force_offline=force_offline), conversations, suite_config),
        run_agent_benchmark("Advanced", AdvancedAgent(suite_config, force_offline=force_offline), conversations, suite_config),
        run_agent_benchmark(
            "Advanced (no compact)",
            AdvancedAgent(no_compact_config, force_offline=force_offline),
            conversations,
            no_compact_config,
        ),
    ]
    print(f"\n## {title}\n")
    print(format_rows(rows))
    print(f"\n{_compare(rows[0], rows[1])}")
    return rows


def print_prompt_trace(dataset: Path, config: LabConfig, checkpoints: tuple[int, ...] = (1, 4, 8, 12, 16)) -> None:
    """Prompt tokens of a single turn, per agent, at a few points of the long thread."""

    conv = load_conversations(dataset)[0]
    force_offline = not config.live
    agents = {
        "Baseline": BaselineAgent(_fresh_config(config, "trace-baseline"), force_offline=force_offline),
        "Advanced": AdvancedAgent(_fresh_config(config, "trace-advanced"), force_offline=force_offline),
    }
    per_turn: dict[str, list[int]] = {name: [] for name in agents}
    for name, agent in agents.items():
        for turn in conv["turns"]:
            per_turn[name].append(agent.reply(conv["user_id"], "trace", turn)["prompt_tokens"])

    points = [c for c in checkpoints if c <= len(conv["turns"])]
    print("\nPrompt tokens of one turn along the stress thread:\n")
    print("| Turn | " + " | ".join(str(c) for c in points) + " |")
    print("|------|" + "|".join("-----" for _ in points) + "|")
    for name, values in per_turn.items():
        print(f"| {name} | " + " | ".join(f"{values[c - 1]:,}" for c in points) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 17 memory benchmark: Baseline vs Advanced")
    parser.add_argument("--live", action="store_true", help="call the configured LLM instead of offline mode")
    args = parser.parse_args()

    config = load_config(Path(__file__).resolve().parent.parent)
    if args.live:
        config = dataclasses.replace(config, live=True)
    mode = f"live ({config.model.provider}/{config.model.model_name})" if config.live else "offline (deterministic)"
    print(f"# Memory benchmark — mode: {mode}")
    print(f"compact threshold = {config.compact_threshold_tokens} tokens, keep = {config.compact_keep_messages} messages")

    run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, "standard")
    run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config, "stress")
    print_prompt_trace(config.data_dir / "advanced_long_context.json", config)


if __name__ == "__main__":
    main()
