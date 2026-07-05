# AgenticAi

LangChain agent with LlamaIndex tools.

## Project structure

```
AgenticAi/
├── src/agentic_ai/
│   ├── main.py                 # CLI entry point
│   ├── agent/                  # LangChain agent setup and runner
│   ├── tools/
│   │   ├── langchain/          # LangChain-native tools
│   │   └── llamaindex/         # LlamaIndex tools wrapped for the agent
│   ├── indexing/               # LlamaIndex docs, indexes, query engines
│   └── config/
│       └── settings.py         # Environment and app settings
├── tests/
├── .env.example
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys
```

## Run locally

```bash
source .venv/bin/activate
cp .env.example .env   # add OPENAI_API_KEY
```

**Interactive chat (recommended):**

```bash
agentic-ai --interactive
# or
python -m agentic_ai.main -i
```

Then paste BOMs at the prompt, for example:

```
You: Please validate this BOM configuration: [Part A, Part B, Part C]
```

Type `quit` or press Ctrl+C to exit.

**Run the 3 built-in test cases:**

```bash
agentic-ai --test
# or
python -m agentic_ai.main --test
```

Add `--quiet` to hide tool-call logs.

Run unit tests (integration test skipped unless `OPENAI_API_KEY` is set):

```bash
pytest
```

## Where to add your code

| Module | Purpose |
|--------|---------|
| `agentic_ai/agent/` | Build and invoke the LangChain agent |
| `agentic_ai/tools/langchain/` | Custom LangChain tools |
| `agentic_ai/tools/llamaindex/` | LlamaIndex tools adapted for LangChain |
| `agentic_ai/indexing/` | Document loading, indexing, RAG pipelines |
| `agentic_ai/main.py` | Wire everything together for the CLI |
