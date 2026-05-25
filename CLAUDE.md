# Audio Transcription — Claude Context

## What this project does
Three-step CLI pipeline: download YouTube audio → transcribe with Whisper → format to readable text.

## Setup
```bash
uv sync                  # install Python deps
brew install yt-dlp      # or: pip install yt-dlp
```

## Common commands
```bash
# Full pipeline — faster-whisper (default)
python main.py --url "https://www.youtube.com/watch?v=<ID>"

# Full pipeline — mlx-whisper (Apple Silicon)
python main.py --url "https://www.youtube.com/watch?v=<ID>" --transcriber v2 --config configs/mlx_whisper.json

# Individual steps
python audio_extractor_01.py -u <URL> -o audio
python transcribe_advanced_01.py data/audio.mp3 --config configs/preferred.json
python transcribe_mlx.py data/audio.mp3 --config configs/mlx_whisper.json
python format_transcription_01.py
```

## Project conventions
- Script names follow the pattern `<purpose>_01.py` — increment the suffix for new versions rather than editing in place. Exception: `transcribe_mlx.py` uses a descriptive suffix to distinguish the engine.
- All generated files go in `data/` (not committed).
- Transcription parameters live in `configs/*.json`, not hardcoded in scripts.
- `main.py` is the orchestrator — it calls the three scripts as subprocesses and exits on first failure.

## Key files
| File | Role |
|---|---|
| `main.py` | Pipeline orchestrator (`--transcriber v1/v2` to switch engine) |
| `audio_extractor_01.py` | YouTube → MP3 via yt-dlp |
| `transcribe_advanced_01.py` | MP3 → JSON via faster-whisper (v1) |
| `transcribe_mlx.py` | MP3 → JSON via mlx-whisper, Apple Silicon only (v2) |
| `transcribe_basic_01.py` | Simpler transcription (no config file) |
| `format_transcription_01.py` | JSON → formatted .txt |
| `configs/preferred.json` | Default config for faster-whisper |
| `configs/mlx_whisper.json` | Default config for mlx-whisper |

## Dependencies
- `faster-whisper`, `mlx-whisper`, `numpy` — managed by uv (`pyproject.toml`)
- `yt-dlp` — external, must be in PATH
- faster-whisper: CUDA GPU recommended; set `"device": "cpu"` in the config to run on CPU
- mlx-whisper: requires Apple Silicon (M1/M2/M3)

## No test suite
There are no automated tests. Validate changes by running the pipeline end-to-end on a short YouTube video.
