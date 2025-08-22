#!/usr/bin/env python3
"""
Sequential Script Executor
Executes three Python scripts in sequence with their respective commands.
"""

import subprocess
import sys
import os
import argparse
from typing import List, Tuple

def run_command(command: List[str], script_name: str) -> bool:
    """
    Execute a command and return True if successful, False otherwise.
    
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
        # Run the command and capture output
        result = subprocess.run(
            command, 
            check=True, 
            text=True, 
            capture_output=True
        )
        
        # Print stdout if there's any output
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        print(f"✓ {script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error executing {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False
    
    except FileNotFoundError:
        print(f"✗ Error: Script '{command[1]}' not found!")
        print("Make sure the script exists in the current directory.")
        return False
    
    except Exception as e:
        print(f"✗ Unexpected error executing {script_name}: {e}")
        return False

def check_script_exists(script_path: str) -> bool:
    """Check if a script file exists."""
    return os.path.isfile(script_path)

def main():
    """Main execution function."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Execute three Python scripts sequentially for audio processing pipeline"
    )
    parser.add_argument(
        "--url", 
        required=True,
        help="YouTube URL to process (e.g., https://www.youtube.com/watch?v=PsjftmuCXxc&t=372s)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    print("Sequential Script Executor")
    print("=" * 60)
    print(f"YouTube URL: {args.url}")
    
    # Define the commands for each script (now using the URL argument)
    commands = [
        (
            ["python", "audio_extractor_01.py", "-u", args.url],
            "Audio Extractor"
        ),
        (
            ["python", "transcribe_advanced_01.py", "data/audio.mp3", "--config", "configs/preferred.json"],
            "Audio Transcriber"
        ),
        (
            ["python", "format_transcription_01.py"],
            "Transcription Formatter"
        )
    ]
    
    # Check if all scripts exist before starting
    scripts_to_check = [
        "audio_extractor_01.py",
        "transcribe_advanced_01.py", 
        "format_transcription_01.py"
    ]
    
    print("Checking if all scripts exist...")
    missing_scripts = []
    for script in scripts_to_check:
        if not check_script_exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"✗ Missing scripts: {', '.join(missing_scripts)}")
        print("Please ensure all scripts are in the current directory.")
        sys.exit(1)
    
    print("✓ All scripts found!")
    
    # Execute each command in sequence
    for i, (command, script_name) in enumerate(commands, 1):
        success = run_command(command, f"Script {i}: {script_name}")
        
        if not success:
            print(f"\n✗ Pipeline failed at step {i} ({script_name})")
            print("Stopping execution.")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All scripts completed successfully!")
    print("Pipeline execution finished.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    
# Example : python main.py --url https://www.youtube.com/watch?v=PsjftmuCXxc&t