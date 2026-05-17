# Audio Transcription — Agent Guide

## Project summary
Python pipeline to transcribe YouTube videos: download audio (yt-dlp), transcribe (faster-whisper), format output.

## Environment setup
```bash
uv sync          # creates .venv and installs faster-whisper + numpy
# yt-dlp must be installed separately and available in PATH
```
Python 3.13+ required (enforced by pyproject.toml).

## Running the pipeline
```bash
python main.py --url "<youtube-url>"
```
Runs the three scripts below in sequence; aborts on first failure.

## Script responsibilities
1. `audio_extractor_01.py` — calls `yt-dlp` to download audio as MP3 into `data/`
2. `transcribe_advanced_01.py` — loads a JSON config, runs faster-whisper, writes `data/audio_transcription.json`
3. `format_transcription_01.py` — reads the JSON, groups segments into paragraphs, writes `data/formatted_transcription.txt`

## Configuration system
All Whisper parameters (model size, device, beam size, VAD, word timestamps…) are defined in `configs/*.json`.
Do not hardcode parameters in scripts — add or edit a config file instead.

| Config | Model | Use case |
|---|---|---|
| `fast.json` | small / int8 | Quick drafts |
| `default.json` | medium / float16 | General use |
| `preferred.json` | large-v3 / int8 | Default for main.py |
| `high_quality.json` | large-v3 / float16 | Maximum accuracy |

## Naming convention
Scripts are versioned by suffix: `audio_extractor_01.py`, `_02.py`, etc.
Create a new numbered file rather than modifying an existing script when making significant changes.

## Data directory
`data/` holds all generated files (MP3, JSON, TXT). Do not commit files from this directory.

## No tests
Verify changes by running the full pipeline on a short video. Check that:
- `data/audio.mp3` is created
- `data/audio_transcription.json` contains valid segments
- `data/formatted_transcription.txt` is readable
