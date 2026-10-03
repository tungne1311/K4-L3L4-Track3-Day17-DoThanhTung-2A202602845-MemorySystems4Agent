# Source

Completed implementation of the Day 17 lab. Results and analysis: [`../REPORT.md`](../REPORT.md).

| File | What it does |
|---|---|
| `model_provider.py` | Provider aliases + `build_chat_model()` for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter` |
| `config.py` | `LabConfig` / `load_config()` from env or `.env` (`LAB_MODE=live` to call real models) |
| `memory_store.py` | Token estimate, `UserProfileStore` (`User.md` with Facts + Changelog), fact extraction with confidence threshold, compact memory |
| `agent_baseline.py` | Agent A: full thread history every turn, nothing persists across threads |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact; live mode uses `create_agent` + tools + `SummarizationMiddleware` |
| `benchmark.py` | Standard + Long-Context Stress tables, a no-compact ablation row, and a per-turn prompt trace |
| `test_agents.py` | 7 tests: `User.md` I/O, compact trigger, cross-session recall, prompt reduction, summary quality, corrections, noise/hedges |

Run from the repo root:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Both run offline (deterministic, no API key). State is written to `state/` (git-ignored).
