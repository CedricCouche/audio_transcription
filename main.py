#!/usr/bin/env python3
"""
Sequential Script Executor
Executes the audio transcription pipeline scripts in sequence.
"""

import subprocess
import sys
import os
import argparse
from typing import List


def run_command(command: List[str], script_name: str) -> bool:
    """
    Execute a command, streaming its output in real time.

    Args:
        command: List of command arguments
        script_name: Name of the script for logging purposes

    Returns:
        bool: True if command executed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Executing {script_name}...")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")

    try:
        subprocess.run(command, check=True)
        print(f"\n✓ {script_name} completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error executing {script_name} (exit code {e.returncode})")
        return False

    except FileNotFoundError:
        print(f"✗ Script not found: {command[1]}")
        print("Make sure the script exists in the current directory.")
        return False

    except Exception as e:
        print(f"✗ Unexpected error executing {script_name}: {e}")
        return False


def check_script_exists(script_path: str) -> bool:
    """Check if a script file exists."""
    return os.path.isfile(script_path)


def main():
    parser = argparse.ArgumentParser(
        description="Execute the audio transcription pipeline for a YouTube URL"
    )
    parser.add_argument(
        "--url", "-u",
        help="YouTube URL to process (required unless --skip-download is set)"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/preferred.json",
        help="Whisper config file to use (default: configs/preferred.json)"
    )
    parser.add_argument(
        "--transcriber",
        choices=["v1", "v2"],
        default="v1",
        help="Transcription engine: v1=faster-whisper (default), v2=mlx-whisper"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and use an existing data/audio.mp3"
    )
    args = parser.parse_args()

    if not args.skip_download and not args.url:
        parser.error("--url is required unless --skip-download is set")

    # Fixed paths shared across steps
    audio_path = "data/audio.mp3"
    json_path = "data/audio_transcription.json"
    txt_path = "data/formatted_transcription.txt"
    
    # Select transcriber script
    transcriber_script = "transcribe_advanced_01.py" if args.transcriber == "v1" else "transcribe_mlx.py"

    print("Audio Transcription Pipeline")
    print("=" * 60)
    if args.url:
        print(f"YouTube URL: {args.url}")
    print(f"Config:      {args.config}")
    print(f"Transcriber: {args.transcriber} ({transcriber_script})")
    if args.skip_download:
        print("Download:    skipped")

    # Validate scripts before starting
    scripts_to_check = [
        "audio_extractor_01.py",
        transcriber_script,
        "format_transcription_01.py",
    ]

    print("\nChecking scripts...")
    missing = [s for s in scripts_to_check if not check_script_exists(s)]
    if missing:
        print(f"✗ Missing scripts: {', '.join(missing)}")
        sys.exit(1)
    print("✓ All scripts found!")

    # Build the command list
    commands = []

    if not args.skip_download:
        commands.append((
            ["python", "audio_extractor_01.py", "-u", args.url, "-o", "audio"],
            "Audio Extractor"
        ))

    commands.append((
        ["python", transcriber_script, audio_path, "--config", args.config],
        f"Audio Transcriber ({args.transcriber})"
    ))

    commands.append((
        ["python", "format_transcription_01.py", "--input", json_path, "--output", txt_path],
        "Transcription Formatter"
    ))

    # Run each step
    for i, (command, script_name) in enumerate(commands, 1):
        success = run_command(command, f"Step {i}: {script_name}")
        if not success:
            print(f"\n✗ Pipeline failed at step {i} ({script_name}). Stopping.")
            sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Audio:       {audio_path}")
    print(f"  Transcript:  {json_path}")
    print(f"  Formatted:   {txt_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# Example:
# python main.py --url "https://www.youtube.com/watch?v=PsjftmuCXxc"
# python main.py --url "https://www.youtube.com/watch?v=PsjftmuCXxc" --config configs/fast.json
# python main.py --url "https://www.youtube.com/watch?v=PsjftmuCXxc" --transcriber v2
# python main.py --skip-download --config configs/high_quality.json
# python main.py --skip-download --transcriber v2 --config configs/preferred.json
