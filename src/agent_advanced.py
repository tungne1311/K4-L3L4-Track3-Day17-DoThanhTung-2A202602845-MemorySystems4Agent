from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FIELD_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    merge_fact,
    render_recall_answer,
)
from model_provider import build_chat_model

try:  # optional: only needed for live mode; tool type hints are resolved at module scope
    from langchain.tools import ToolRuntime
except ImportError:  # pragma: no cover - offline-only environments
    ToolRuntime = None

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI có bộ nhớ dài hạn. Trả lời bằng tiếng Việt, ngắn gọn và đúng trọng tâm. "
    "Hồ sơ người dùng (User.md) bên dưới là nguồn sự thật mới nhất: luôn ưu tiên nó khi có mâu thuẫn "
    "với tin nhắn cũ, và tuân theo style trả lời trong hồ sơ."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. within-session memory: recent messages of the thread (CompactMemoryManager)
    2. persistent memory: `User.md` per user, survives new threads
    3. compact memory: older messages folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._live_message_count: dict[str, int] = {}

        self.langchain_agent = None
        if self.config.live and not force_offline and self.config.model.is_usable:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def profile_prompt(self, user_id: str) -> str:
        """Only the Facts go into the prompt; the changelog is an audit trail, not context."""

        facts = self.profile_store.facts(user_id)
        if not facts:
            return "User.md: (trống)"
        return "User.md:\n" + "\n".join(f"- {key}: {value}" for key, value in facts.items())

    def _remember(self, user_id: str, message: str) -> list[str]:
        """Steps 1-2: extract confident facts and persist them. Returns changed keys."""

        current = self.profile_store.facts(user_id)
        changed = []
        for key, value in extract_profile_updates(message, self.config.profile_min_confidence).items():
            if self.profile_store.upsert_fact(user_id, key, merge_fact(key, current.get(key), value)):
                changed.append(key)
        return changed

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        response = self._offline_response(user_id, thread_id, message)
        if changed and not render_recall_answer(message, {}):
            facts = self.profile_store.facts(user_id)
            response += " Cập nhật User.md: " + "; ".join(f"{FIELD_LABELS[k]} = {facts[k]}" for k in changed) + "."
        self.compact_memory.append(thread_id, "assistant", response)

        agent_tokens = estimate_tokens(message) + estimate_tokens(response)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
            "mode": "offline",
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Context carried into one turn: system + User.md facts + compact summary + kept messages."""

        ctx = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(SYSTEM_PROMPT)
            + estimate_tokens(self.profile_prompt(user_id))
            + estimate_tokens(str(ctx["summary"]))
            + self.compact_memory.message_tokens(thread_id)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Deterministic answer: recall questions are answered from User.md."""

        answer = render_recall_answer(message, self.profile_store.facts(user_id))
        return answer or "Đã ghi nhận."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # Persist deterministically too, so User.md never depends on the model choosing to call a tool.
        changed = self._remember(user_id, message)
        self.compact_memory.append(thread_id, "user", message)

        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": f"advanced:{thread_id}"}},
            context=context,
        )
        start = self._live_message_count.get(thread_id, 0)
        new_messages = result["messages"][start:] if start <= len(result["messages"]) else result["messages"]
        self._live_message_count[thread_id] = len(result["messages"])

        prompt_tokens = output_tokens = 0
        for msg in new_messages:
            usage = getattr(msg, "usage_metadata", None) or {}
            prompt_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        response = result["messages"][-1].content
        if isinstance(response, list):
            response = "".join(part.get("text", "") for part in response if isinstance(part, dict))

        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(message) + (output_tokens or estimate_tokens(response))
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: model + InMemorySaver + User.md tools + dynamic prompt + summarization."""

        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRequest, SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        if ToolRuntime is None:
            return None

        store = self.profile_store
        agent = self

        @tool
        def read_user_profile(runtime: ToolRuntime[AgentContext]) -> str:
            """Đọc toàn bộ User.md (hồ sơ bền vững) của người dùng hiện tại."""

            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Lưu hoặc cập nhật một fact ổn định vào User.md.

            key: một trong name, location, profession, style, interests, favorite_drink, favorite_food, pet.
            Chỉ lưu fact người dùng khẳng định chắc chắn; không lưu câu hỏi, câu đùa hay thông tin tạm thời.
            """

            if key not in FIELD_LABELS:
                return f"Bỏ qua: key không hợp lệ ({key})."
            user_id = runtime.context.user_id
            old = store.facts(user_id).get(key)
            changed = store.upsert_fact(user_id, key, merge_fact(key, old, value))
            return "Đã cập nhật User.md." if changed else "Không có thay đổi."

        @dynamic_prompt
        def inject_profile(request: ModelRequest) -> str:
            return f"{SYSTEM_PROMPT}\n\n{agent.profile_prompt(request.runtime.context.user_id)}"

        model = build_chat_model(self.config.model)
        return create_agent(
            model=model,
            tools=[read_user_profile, save_user_fact],
            middleware=[
                inject_profile,
                SummarizationMiddleware(
                    model=model,
                    trigger=("tokens", self.config.compact_threshold_tokens),
                    keep=("messages", self.config.compact_keep_messages),
                ),
            ],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
