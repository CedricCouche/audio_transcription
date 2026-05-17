# Audio Transcription

A pipeline that downloads audio from a YouTube video, transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and formats the result into readable text.

## Pipeline

```
YouTube URL → audio_extractor_01.py → transcribe_advanced_01.py → format_transcription_01.py
                 (yt-dlp / MP3)           (Whisper / JSON)              (formatted .txt)
```

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) installed and in your PATH
- A CUDA-capable GPU (recommended) or CPU fallback

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
python main.py --url "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

This runs all three steps in sequence and stops on the first failure. Output files land in `data/`.

### Individual scripts

**1. Download audio**
```bash
python audio_extractor_01.py --url "https://www.youtube.com/watch?v=<VIDEO_ID>" --output audio
```

**2. Transcribe**
```bash
python transcribe_advanced_01.py data/audio.mp3 --config configs/preferred.json
```
Produces `data/audio_transcription.json`.

**3. Format**
```bash
python format_transcription_01.py
```
Reads `data/audio_transcription.json`, writes `data/formatted_transcription.txt`.

## Configuration profiles

| Profile | Model | Compute | Speed | Quality |
|---|---|---|---|---|
| `fast.json` | small | int8 | Fastest | Lower |
| `default.json` | medium | float16 | Balanced | Good |
| `preferred.json` | large-v3 | int8 | Slower | High + word timestamps |
| `high_quality.json` | large-v3 | float16 | Slowest | Highest + word timestamps |

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
├── transcribe_basic_01.py     # Simpler transcription alternative
├── format_transcription_01.py # JSON → formatted text
├── configs/
│   ├── fast.json
│   ├── default.json
│   ├── preferred.json
│   └── high_quality.json
└── data/                      # Generated files
```
