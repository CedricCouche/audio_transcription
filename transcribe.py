#!/usr/bin/env python3
"""
Audio transcription using faster-whisper
Requires: pip install faster-whisper
"""

from faster_whisper import WhisperModel
import time
import sys
import os

def transcribe_audio(audio_path, model_size="large-v3", device="auto", compute_type="float16"):
    """
    Transcribe audio file using faster-whisper
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large-v2", "large-v3")
        device (str): Device to use ("cpu", "cuda", "auto")
        compute_type (str): Computation precision ("float16", "int8", "float32")
    
    Returns:
        str: Transcribed text
    """
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return None
    
    print(f"Loading Whisper model: {model_size}")
    print(f"Device: {device}, Compute type: {compute_type}")
    
    # Initialize the model
    # With 32GB RAM, you can use large models without issues
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing: {audio_path}")
    start_time = time.time()
    
    # Transcribe with timestamps
    segments, info = model.transcribe(
        audio_path, 
        beam_size=5,  # Higher beam size for better accuracy
        language="en",  # Set to None for auto-detection, "fr" for French, "en" for English
        task="transcribe"  # or "translate" to translate to English
    )
    
    # Collect all text segments
    full_transcription = []
    detailed_transcription = []
    
    print("\n--- Transcription with timestamps ---")
    for segment in segments:
        timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
        text = segment.text.strip()
        
        print(f"{timestamp} {text}")
        
        full_transcription.append(text)
        detailed_transcription.append(f"{timestamp} {text}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n--- Summary ---")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Speed: {info.duration/elapsed_time:.2f}x real-time")
    
    # Return full text
    return " ".join(full_transcription), detailed_transcription


def save_transcription(text, detailed_text, audio_path):
    """Save transcription to text files"""
    base_name = os.path.splitext(audio_path)[0]
    
    # Format text with line breaks after sentences
    formatted_text = format_text_with_line_breaks(text)
    
    # Save simple transcription
    with open(f"{base_name}_transcription.txt", "w", encoding="utf-8") as f:
        f.write(formatted_text)
    
    # Save detailed transcription with timestamps
    with open(f"{base_name}_detailed_transcription.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(detailed_text))
    
    print(f"\nTranscription saved to:")
    print(f"  - {base_name}_transcription.txt")
    print(f"  - {base_name}_detailed_transcription.txt")


def format_text_with_line_breaks(text):
    """Add line breaks after sentence endings"""
    import re
    
    # Replace sentence endings with line breaks
    # Matches: period, exclamation, question mark followed by space and capital letter
    formatted = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)
    
    # Handle sentence endings at the very end of text
    formatted = re.sub(r'([.!?])$', r'\1\n', formatted)
    
    # Clean up any double line breaks
    formatted = re.sub(r'\n\n+', '\n\n', formatted)
    
    return formatted.strip()


if __name__ == "__main__":
    # Configuration
    AUDIO_FILE = "Audio_01.mp3"  # Change this to your MP3 file path
    MODEL_SIZE = "medium"  # Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    
    # For 32GB RAM, you can use:
    # - "large-v3" for best accuracy
    # - "medium" for faster processing
    # - Use "cuda" if you have a compatible GPU, otherwise "cpu"
    DEVICE = "cuda"  # "cpu", "cuda", or "auto"
    COMPUTE_TYPE = "int8"  # "float16" for GPU, "int8" for CPU optimization
    
    # Allow command line argument for audio file
    if len(sys.argv) > 1:
        AUDIO_FILE = sys.argv[1]
    
    print("=== Faster Whisper Audio Transcription ===")
    print(f"Audio file: {AUDIO_FILE}")
    
    # Transcribe
    transcription, detailed = transcribe_audio(
        AUDIO_FILE, 
        model_size=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    
    if transcription:
        print(f"\n--- Full Transcription ---")
        print(transcription)
        
        # Save to files
        save_transcription(transcription, detailed, AUDIO_FILE)
    else:
        print("Transcription failed.")

# Example usage:
# python transcribe.py your_audio.mp3