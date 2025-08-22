#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

def download_audio(url, output_filename):
    """
    Download audio from a YouTube URL using yt-dlp
    
    Args:
        url (str): YouTube URL to download audio from
        output_filename (str): Custom filename
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        print(f"Starting download for: {url}")
        
        # Set output path
        output_path = f'data/{output_filename}.%(ext)s'

        result = subprocess.run([
            'yt-dlp', 
            '-x', 
            '--audio-format', 
            'mp3',
            '-o',
            output_path,
            url
        ], capture_output=True, text=True, check=True)
        
        print("Success!")
        print("Output:", result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False
        
    except FileNotFoundError:
        print("Error: yt-dlp not found. Make sure it's installed and in your PATH.")
        print("Install with: pip install yt-dlp")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download audio from YouTube URL using yt-dlp')
    parser.add_argument('--url', '-u', required=True, help='YouTube URL to download audio from')
    parser.add_argument('--output', '-o', default='audio', help='Output filename (without extension, default: audio)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Download audio with provided URL and output filename
    success = download_audio(args.url, args.output)
    
    if success:
        print("\nDownload completed successfully!")
        print("File saved in 'data' folder")
        sys.exit(0)
    else:
        print("\nDownload failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()