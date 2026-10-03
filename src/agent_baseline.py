from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates, merge_fact, render_recall_answer
from model_provider import build_chat_model

SYSTEM_PROMPT = "Bạn là trợ lý AI. Trả lời bằng tiếng Việt, ngắn gọn và đúng trọng tâm."


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0
    facts: dict[str, str] = field(default_factory=dict)  # only what was said in THIS thread
    live_message_count: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Every turn re-sends the full thread history (no compaction).
    - No `User.md`: a new thread id starts from zero.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None
        if self.config.live and not force_offline and self.config.model.is_usable:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # Prompt = system + the whole thread so far: grows linearly per turn, quadratically in total.
        prompt_tokens = estimate_tokens(SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

        for key, value in extract_profile_updates(message, self.config.profile_min_confidence).items():
            session.facts[key] = merge_fact(key, session.facts.get(key), value)

        response = render_recall_answer(message, session.facts) or "Đã ghi nhận trong cuộc trò chuyện này."
        session.messages.append({"role": "assistant", "content": response})

        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens, "mode": "offline"}

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": f"baseline:{thread_id}"}},
        )
        new_messages = result["messages"][session.live_message_count:]
        session.live_message_count = len(result["messages"])

        prompt_tokens = output_tokens = 0
        for msg in new_messages:
            usage = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        response = result["messages"][-1].content
        if isinstance(response, list):  # some providers return content blocks
            response = "".join(part.get("text", "") for part in response if isinstance(part, dict))

        session.messages += [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        agent_tokens = estimate_tokens(message) + (output_tokens or estimate_tokens(response))
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "agent_tokens": agent_tokens, "prompt_tokens": prompt_tokens, "mode": "live"}

    def _maybe_build_langchain_agent(self):
        """`create_agent` + `InMemorySaver`: short-term memory per thread_id, nothing else."""

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
