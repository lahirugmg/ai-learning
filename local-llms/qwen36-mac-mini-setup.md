# Qwen 3.6 35B-A3B on Mac mini: Action Log

**Goal:** Run Qwen3.6-35B-A3B locally on the Mac mini as an efficient, always-on model server with an OpenAI-compatible API.
**Approach chosen:** Step 1, a headless MLX server (`mlx-lm`)
**Started:** 2026-10-30

---

## Status

| # | Step | Status |
|---|------|--------|
| 0 | Check Mac mini hardware (chip, RAM, free disk) | ☑ Done |
| 1 | Install headless MLX server (`mlx-lm`) | ☑ Done |
| 2 | Download the model weights | ☑ Done |
| 3 | Start the server and smoke-test it | ☑ Done |
| 4 | Expose the server on the LAN (optional) | ☐ To do |
| 5 | Run it as an always-on service (launchd) | ☐ To do |
| 6 | Tune GPU memory limit (optional) | ☐ To do |
| 7 | Connect clients (agent, coding tools) | ◐ Started: Python scripts only |

---

## Step 0: Check hardware

Run on the Mac mini:

```bash
sysctl -n machdep.cpu.brand_string                          # chip
echo "$(($(sysctl -n hw.memsize)/1024/1024/1024)) GB RAM"   # unified memory
df -h ~                                                      # free disk (need ~25 GB+)
sw_vers                                                      # macOS version
```

Use the RAM to choose a quant:

| RAM | Model to use | Notes |
|-----|--------------|-------|
| 16 GB | `mlx-community/Qwen3.6-35B-A3B-4bit` (or 3-bit) | Tight. Close other apps and keep context small. |
| 24–32 GB | `mlx-community/Qwen3.6-35B-A3B-4bit` | Comfortable choice. |
| 48 GB+ | `...-6bit` or `...-8bit` | Better quality with more room for context. |

**Recorded:** Chip: Apple M6  RAM: 32 GB  Free disk: 13 GB of 228 GB after the download (94% used)  macOS: 27.2

---

## Step 1: Install the MLX server

```bash
# Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install uv
uv tool install mlx-lm
uv tool update-shell        # adds ~/.local/bin to PATH; then open a new terminal

which mlx_lm.server         # verify
```

> For image input later, install `mlx-vlm` as well: `uv tool install "mlx-vlm[torch]"`

---

## Step 2: Download the model (about 20 GB for 4-bit)

Downloading first means the server's first start won't stall on the download.

```bash
uvx --from huggingface_hub hf download mlx-community/Qwen3.6-35B-A3B-4bit
```

Weights are saved to `~/.cache/huggingface/hub/`.

> The model's own folder holds mostly symlinks, so `du` reports only a few MB for it. The real files are in the shared store: `du -sh ~/.cache/huggingface/hub/blobs` (19 GB).

---

## Step 3: Start the server and smoke-test it

```bash
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
```

In a second terminal:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": false}
      }'
```

> Thinking is on by default and counts toward `max_tokens`. Without the `chat_template_kwargs` line, this test uses all 100 tokens on thinking and returns `"content": null` with `"finish_reason": "length"`, which looks like a failure even though the server is fine.

**Recorded:** Works? ☑  Tokens/sec: about 55–65 output (see [measured speed](#measured-speed))  Memory used: 19 GB after loading, 22 GB peak with a 9,000-token prompt (`footprint -p <pid>`, because `ps` leaves out most of this memory)

---

## Step 4: Expose the server on the LAN (optional)

By default the server only listens on localhost. To reach it from the MacBook Pro or other machines:

```bash
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --host 0.0.0.0 --port 8080
```

- Endpoint: `http://<mac-mini-ip>:8080/v1`
- ⚠️ The server has **no authentication**. Keep it on the home LAN only and never port-forward it. For remote access, use Tailscale.
- Give the mini a DHCP reservation on the router so its IP doesn't change.

---

## Step 5: Run it as an always-on service (launchd)

Create `~/Library/LaunchAgents/com.local.mlx-qwen.plist`, replacing `USERNAME` with your macOS user name:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.local.mlx-qwen</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/USERNAME/.local/bin/mlx_lm.server</string>
    <string>--model</string><string>mlx-community/Qwen3.6-35B-A3B-4bit</string>
    <string>--host</string><string>0.0.0.0</string>
    <string>--port</string><string>8080</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/mlx-qwen.log</string>
  <key>StandardErrorPath</key><string>/tmp/mlx-qwen.err</string>
</dict>
</plist>
```

> ⚠️ This plist binds to `0.0.0.0`, so it exposes the server to the LAN even if you skipped Step 4. To keep it on this machine only, remove the `--host` line.
>
> Stop any server running in a terminal first. Both would need port 8080, and `KeepAlive` would keep restarting the service.

```bash
launchctl load ~/Library/LaunchAgents/com.local.mlx-qwen.plist
launchctl list | grep mlx-qwen          # confirm it's running
tail -f /tmp/mlx-qwen.err               # watch logs
# To stop: launchctl unload ~/Library/LaunchAgents/com.local.mlx-qwen.plist
```

Also set these so the mini stays awake:
- System Settings → Energy → turn on **Prevent automatic sleeping** and **Start up automatically after a power failure**
- Turn on auto-login (or move the service to a LaunchDaemon) so it starts after a reboot

---

## Step 6: Tune GPU memory limit (optional)

By default, macOS limits how much unified memory the GPU can use. On a dedicated mini you can raise that limit, leaving about 4–6 GB for the OS:

```bash
sudo sysctl iogpu.wired_limit_mb=26000   # example for 32 GB
```

This resets on reboot, so add it to a startup script if it helps.

---

## Step 7: Connect clients

Any tool that speaks the OpenAI API can use the server:
- Base URL: `http://<mac-mini-ip>:8080/v1`, or `http://127.0.0.1:8080/v1` on the mini itself
- API key: any placeholder string, e.g. `local`
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`

Candidates: personal agent (OpenClaw / Hermes), aider, opencode, scripts.

Scripts in this folder, run from here with `uv run chat.py "Why is the sky blue?"` or `uv run benchmark.py`:
- [chat.py](chat.py): a minimal API call with streaming
- [benchmark.py](benchmark.py): measures time to first token and output speed

### What the server supports

Tested on 2026-10-30 with `mlx-lm` 0.32.2.

| Feature | Result | Notes |
|---------|--------|-------|
| Chat completions and streaming | ✅ Works | |
| Tool calling (`tools`) | ✅ Works | Returned a correct function call with `finish_reason: tool_calls` |
| Prompt caching | ✅ Works | Reused tokens appear in `usage.prompt_tokens_details.cached_tokens` |
| Turning off thinking | ✅ Works | Add `"chat_template_kwargs": {"enable_thinking": false}` to the request |
| Health check | ✅ Works | `GET /health` |
| Embeddings (`/v1/embeddings`) | ❌ 404 | Needs a separate embedding model, such as `bge-small-en-v1.5` through `fastembed` |
| JSON mode (`response_format`) | ❌ Ignored | Returned plain prose with no error. Ask for JSON in the prompt and validate it in code |

Defaults worth knowing:
- **Thinking is on.** The reasoning arrives in a separate `reasoning` field and the answer in `content`. A one-sentence answer took 376 tokens with thinking and 41 without.
- **`max_tokens` is 512** if the request doesn't set it, so long answers with thinking on can get cut off. Check `finish_reason` for `length`.
- **`temperature` is 0.0**, so replies vary little between runs unless you raise it.

To turn thinking off for every request, start the server with `--chat-template-args '{"enable_thinking":false}'`. This comes from `mlx_lm.server --help` and hasn't been tested here.

### Measured speed

4-bit model, thinking off, one run per prompt size, from [benchmark.py](benchmark.py):

| Prompt tokens | Time to first token | Output speed |
|---------------|---------------------|--------------|
| 109 | 0.37 s | 64.6 tokens/s |
| 636 | 0.54 s | 64.1 tokens/s |
| 2,385 | 1.86 s | 62.0 tokens/s |
| 9,387 | 6.25 s | 59.5 tokens/s |

Longer prompts mostly add waiting before the first token, at roughly 1,500 prompt tokens per second. Output speed barely changes. A second run the same day measured 54–57 tokens/s, so treat these numbers as rough.

---

## Fallback options (if MLX server doesn't fit)

- **Ollama:** `brew install ollama`, then `OLLAMA_USE_MLX=1 ollama serve`, then `ollama pull qwen3.6:35b-a3b`. This is a convenient background service with wide tool support. Its MLX backend is meant for Macs with 32 GB or more.
- **LM Studio:** a GUI app that automatically uses the MLX build. The app has to stay open.

---

## Notes / log

- 2026-10-30: Chose the headless MLX server as step 1.
- 2026-10-30: Steps 0–3 done. The server runs in a terminal window for now, so closing that window stops it (Step 5 fixes this).
- 2026-10-30: The original smoke test returned no content because thinking used all 100 tokens. Fixed by turning thinking off in the request.
- 2026-10-30: Tested the API. Tool calling works; embeddings and JSON mode don't.
- 2026-10-30: After the download, the disk is 94% full with 13 GB free, and about 4 GB of swap is in use. Larger quants won't fit until space is freed.
