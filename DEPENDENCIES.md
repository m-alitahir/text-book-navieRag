# Dependencies

## Models (served locally via Ollama)

| Purpose | Model |
|---|---|
| Embeddings | `qwen3-embedding:4b` |
| Chat / answer generation (also used for query cleanup) | `qwen3:8b` |

Pull both before running the project:
```
ollama pull qwen3-embedding:4b
ollama pull qwen3:8b
```
Ollama must be running locally (`ollama serve`, or the desktop app) on the default port — the code talks to it at `http://localhost:11434`.

## Python libraries

| Library | Used for |
|---|---|
| `gradio` | Browser-based chat UI |
| `requests` | HTTP calls to the local Ollama API (embeddings + chat) |
| `pypdf` | Extracting text from the textbook PDF |

Install with:
```
pip install gradio requests pypdf
```

No API keys, cloud accounts, or `.env` file are required — everything runs locally against Ollama.
