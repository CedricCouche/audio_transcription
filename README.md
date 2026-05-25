# Audio Transcription

A pipeline that downloads audio from a YouTube video, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) or [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon), and formats the result into readable text.

## Pipeline

```
YouTube URL → audio_extractor_01.py → transcribe_advanced_01.py  → format_transcription_01.py
                 (yt-dlp / MP3)          or transcribe_mlx.py          (formatted .txt)
                                           (Whisper / JSON)
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed and in your PATH
- A CUDA-capable GPU for faster-whisper, or Apple Silicon for mlx-whisper

## Installation

```bash
uv sync
```

Install `yt-dlp` separately if not already available:

```bash
pip install yt-dlp
# or: brew install yt-dlp
```

## Usage

### Full pipeline

```bash
# faster-whisper (default)
python main.py --url "https://www.youtube.com/watch?v=<VIDEO_ID>"

# mlx-whisper (Apple Silicon)
python main.py --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --transcriber v2 --config configs/mlx_whisper.json
```

Use `--skip-download` to skip the download step and reuse an existing `data/audio.mp3`.

This runs all three steps in sequence and stops on the first failure. Output files land in `data/`.

### Individual scripts

**1. Download audio**
```bash
python audio_extractor_01.py --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --output audio
```

**2. Transcribe**
```bash
# faster-whisper
python transcribe_advanced_01.py data/audio.mp3 --config configs/preferred.json

# mlx-whisper (Apple Silicon)
python transcribe_mlx.py data/audio.mp3 --config configs/mlx_whisper.json
```
Produces `data/audio_transcription.json`.

**3. Format**
```bash
python format_transcription_01.py
```
Reads `data/audio_transcription.json`, writes `data/formatted_transcription.txt`.

## Configuration profiles

### faster-whisper

| Profile | Model | Compute | Speed | Quality |
|---|---|---|---|---|
| `fast.json` | small | int8 | Fastest | Lower |
| `default.json` | medium | float16 | Balanced | Good |
| `preferred.json` | large-v3 | int8 | Slower | High + word timestamps |
| `high_quality.json` | large-v3 | float16 | Slowest | Highest + word timestamps |

### mlx-whisper (Apple Silicon only)

| Profile | Model | Notes |
|---|---|---|
| `mlx_whisper.json` | whisper-large-v3 | Runs natively on Apple Silicon via MLX |

Pass any profile with `--config configs/<name>.json`. The pipeline uses `preferred.json` by default.

## Output

- `data/audio.mp3` — downloaded audio
- `data/audio_transcription.json` — full transcription with segments, metadata, and statistics
- `data/formatted_transcription.txt` — human-readable text with timestamps and paragraph grouping

## Project structure

```
.
├── main.py                    # Orchestrates the full pipeline
├── audio_extractor_01.py      # YouTube → MP3 (yt-dlp)
├── transcribe_advanced_01.py  # MP3 → JSON (faster-whisper)
├── transcribe_mlx.py          # MP3 → JSON (mlx-whisper, Apple Silicon)
├── transcribe_basic_01.py     # Simpler transcription alternative
├── format_transcription_01.py # JSON → formatted text
├── configs/
│   ├── fast.json
│   ├── default.json
│   ├── preferred.json
│   ├── high_quality.json
│   └── mlx_whisper.json
└── data/                      # Generated files
```
