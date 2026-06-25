# GrokWeb-to-API SDK

> Unofficial Python SDK for [Grok](https://grok.com/) — OpenAI-compatible interface, no cloud service required.

Drop-in client library for Python. Use it in your code via the `GrokOpenAIClient` wrapper.

---

## Setup

### 1. Clone and install with Poetry

```bash
git clone https://github.com/LimonTH/GrokWeb-to-API.git
cd GrokWeb-to-API
poetry install
```

> If you don't have Poetry:
>
> - **Arch / Manjaro (pacman):** `sudo pacman -S python-poetry`
> - **Ubuntu / Debian (official script):** `curl -sSL https://install.python-poetry.org | python3 -`
> - **Fedora:** `sudo dnf install poetry`
> - **macOS (Homebrew):** `brew install poetry`
> - **Universal:** `pip install poetry`

### 2. Get authentication cookies
1. Open [grok.com](https://grok.com) in your browser and log in.
2. Open DevTools (`F12`) → **Network** tab.
3. Refresh the page and find any request to `grok.com`.
4. Copy the `Cookie` header value from the request headers.

---

## Usage

### A. Direct Python library (`GrokOpenAIClient`)

Drop‑in replacement for `openai.OpenAI.chat.completions.create`:

```python
from grok_client import GrokOpenAIClient

client = GrokOpenAIClient(
    cookies_str="sso=...; sso-rw=...",
    model_name="grok-4.3",
)

# Non-streaming
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    stream=False,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])

# Streaming
stream = client.chat_completion(
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
    temperature=0.9,
)
full = client.process_streaming_response(stream)
```

#### Supported parameters

| Parameter         | Type                     | Description                               |
|-------------------|--------------------------|-------------------------------------------|
| `messages`        | `list[dict[str, str]]`   | Chat history (standard OpenAI format)     |
| `stream`          | `bool`                   | Yield tokens as they arrive               |
| `temperature`     | `float`                  | Sampling temperature (passed to grok.com) |
| `max_tokens`      | `int \| None`            | Maximum tokens to generate                |
| `response_format` | `dict[str, str] \| None` | `{"type": "json_object"}` supported       |
| `functions`       | `list[dict] \| None`     | Function/tool definitions                 |
| `function_call`   | `str \| dict \| None`    | Force a specific function                 |

### B. Interactive chat

```bash
poetry run python -m grok_client.interactive_chat
```

Supports `/temp 0.7`, `/json`, `/system ...` at runtime.

---

## Project structure

```
grok_client/
├── __init__.py            # Exports GrokClient & GrokOpenAIClient
├── client.py              # Low-level HTTP client (talks to grok.com)
└── grok_openai_client.py  # OpenAI-compatible wrapper
```
---

## Disclaimer

This is an unofficial API client. It relies on reverse‑engineering browser requests and may break if grok.com changes its API. Use at your own risk.