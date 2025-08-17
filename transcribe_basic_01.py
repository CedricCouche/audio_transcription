#!/usr/bin/env python3
"""
Basic audio transcription using faster-whisper
Output: JSON file with segments and metadata
"""

from faster_whisper import WhisperModel
import json
import sys
import os
import time

def transcribe_to_json(audio_path, model_size="medium", device="cuda", compute_type="int8"):
    """
    Transcribe audio file and return JSON data
    
    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size
        device (str): Device to use ("cpu", "cuda", "auto")
        compute_type (str): Computation precision
    
    Returns:
        dict: JSON data with transcription results
    """
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return None
    
    print(f"Loading Whisper model: {model_size}")
    print(f"Device: {device}, Compute type: {compute_type}")
    
    # Initialize the model
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Transcribing: {audio_path}")
    start_time = time.time()
    
    # Transcribe
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        language="en",   # Set to None for auto-detection, "fr" for French, "en" for English
        task="transcribe"
    )
    
    # Collect segments
    segments_list = []
    full_text_parts = []
    
    print("\n--- Processing segments ---")
    for i, segment in enumerate(segments):
        segment_data = {
            "id": i,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "duration": round(segment.end - segment.start, 2),
            "text": segment.text.strip()
        }
        
        segments_list.append(segment_data)
        full_text_parts.append(segment.text.strip())
        
        # Show progress
        print(f"Segment {i+1}: [{segment.start:.1f}s-{segment.end:.1f}s] {segment.text.strip()}")
    
    processing_time = time.time() - start_time
    
    # Create JSON structure
    json_data = {
        "metadata": {
            "file": os.path.basename(audio_path),
            "file_path": os.path.abspath(audio_path),
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_seconds": round(info.duration, 2),
            "processing_time_seconds": round(processing_time, 2),
            "processing_speed": round(info.duration / processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "transcription": {
            "full_text": " ".join(full_text_parts),
            "word_count": len(" ".join(full_text_parts).split()),
            "segment_count": len(segments_list)
        },
        "segments": segments_list
    }
    
    print(f"\n--- Summary ---")
    print(f"Language: {info.language} (confidence: {info.language_probability:.3f})")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Speed: {info.duration/processing_time:.2f}x real-time")
    print(f"Segments: {len(segments_list)}")
    print(f"Words: {json_data['transcription']['word_count']}")
    
    return json_data

def save_json(data, audio_path):
    """Save transcription data to JSON file"""
    if not data:
        return None
    
    # Create output filename
    base_name = os.path.splitext(audio_path)[0]
    json_path = f"{base_name}_transcription.json"
    
    # Save JSON with pretty formatting
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON saved to: {json_path}")
    return json_path

if __name__ == "__main__":
    # Configuration
    MODEL_SIZE = "medium"  # "tiny", "base", "small", "medium", "large-v2", "large-v3"
    DEVICE = "cuda"       # "cpu", "cuda", "auto"
    COMPUTE_TYPE = "int8" # "int8", "float16", "float32"
    
    # Get audio file from command line or use default
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "your_audio.mp3"  # Change this default
    
    print("=== Basic Whisper Transcription to JSON ===")
    print(f"Audio file: {audio_file}")
    print(f"Model: {MODEL_SIZE}")
    
    # Transcribe
    result = transcribe_to_json(
        audio_file,
        model_size=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    
    # Save to JSON
    if result:
        json_path = save_json(result, audio_file)
        
        # Show a preview of the JSON structure
        print(f"\n--- JSON Structure Preview ---")
        print("Keys:", list(result.keys()))
        print("Metadata keys:", list(result["metadata"].keys()))
        print("Transcription keys:", list(result["transcription"].keys()))
        print(f"First 3 segments:")
        for segment in result["segments"][:3]:
            print(f"  [{segment['start']}s-{segment['end']}s]: {segment['text'][:50]}...")
    else:
        print("Transcription failed.")