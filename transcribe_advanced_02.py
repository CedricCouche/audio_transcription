#!/usr/bin/env python3
"""
Advanced audio transcription using mlx-whisper with JSON configuration
Output: JSON file with segments and metadata
"""

import mlx.core as mx
import mlx_whisper
import json
import sys
import os
import time
import argparse

KNOWN_PARAMS = {
    'model', 'device', 'language', 'task', 'temperature', 'beam_size',
    'best_of', 'patience', 'length_penalty', 'repetition_penalty', 
    'no_repeat_ngram_size', 'compression_ratio_threshold', 'log_prob_threshold',
    'no_speech_threshold', 'condition_on_previous_text', 'initial_prompt',
    'prefix', 'suppress_blank', 'suppress_tokens', 'without_timestamps',
    'max_initial_timestamp', 'word_timestamps', 'prepend_punctuations',
    'append_punctuations'
}


def load_config(config_path):
    """
    Load configuration from JSON file

    Args:
        config_path (str): Path to JSON configuration file

    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        unknown = set(config) - KNOWN_PARAMS
        if unknown:
            print(f"Warning: unknown config keys will be ignored: {', '.join(sorted(unknown))}")
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return None


def transcribe_advanced(audio_path, **kwargs):
    """
    Advanced transcribe with mlx-whisper
    
    Args:
        audio_path (str): Path to the audio file
        **kwargs: All transcription parameters
    
    Returns:
        dict: JSON data with transcription results
    """
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return None
    
    # Model parameters
    model_name = kwargs.get('model', 'base')
    device = kwargs.get('device', 'gpu')  # mlx-whisper uses 'gpu' by default
    language = kwargs.get('language', None)  # None for auto-detection
    task = kwargs.get('task', 'transcribe')  # 'transcribe' or 'translate'
    
    # Convert device parameter for compatibility
    if device == 'cuda':
        device = 'gpu'
    elif device == 'cpu':
        device = 'cpu'
    
    print(f"Loading Whisper model: {model_name}")
    print(f"Device: {device}")
    
    print(f"Transcribing: {audio_path}")
    
    start_time = time.time()
    
    # Transcribe with mlx-whisper
    # mlx-whisper has a simpler API than faster-whisper
    try:
        result = mlx_whisper.transcribe(
            audio=audio_path,
            path_or_hf_repo=model_name,
            language=language,
            task=task,
            temperature=kwargs.get('temperature', 0.0),
            beam_size=kwargs.get('beam_size', 5),
            best_of=kwargs.get('best_of', 5),
            patience=kwargs.get('patience', 1.0),
            length_penalty=kwargs.get('length_penalty', 1.0),
            repetition_penalty=kwargs.get('repetition_penalty', 1.0),
            no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 0),
            compression_ratio_threshold=kwargs.get('compression_ratio_threshold', 2.4),
            log_prob_threshold=kwargs.get('log_prob_threshold', -1.0),
            no_speech_threshold=kwargs.get('no_speech_threshold', 0.6),
            condition_on_previous_text=kwargs.get('condition_on_previous_text', True),
            initial_prompt=kwargs.get('initial_prompt', None),
            prefix=kwargs.get('prefix', None),
            suppress_blank=kwargs.get('suppress_blank', True),
            suppress_tokens=kwargs.get('suppress_tokens', [-1]),
            without_timestamps=kwargs.get('without_timestamps', False),
            max_initial_timestamp=kwargs.get('max_initial_timestamp', 1.0),
            word_timestamps=kwargs.get('word_timestamps', False),
            prepend_punctuations=kwargs.get('prepend_punctuations', '"\'¿([{-'),
            append_punctuations=kwargs.get('append_punctuations', '"\'.。,!?:)]}'),
        )
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    
    processing_time = time.time() - start_time
    
    # Process segments from mlx-whisper output
    segments_list = []
    full_text_parts = []
    word_level_data = []
    detected_language = language if language else 'unknown'
    duration_seconds = 0.0
    
    if 'segments' in result:
        for i, segment in enumerate(result['segments']):
            segment_data = {
                "id": i,
                "seek": segment.get('seek', 0),
                "start": round(segment.get('start', 0), 3),
                "end": round(segment.get('end', 0), 3),
                "duration": round(segment.get('end', 0) - segment.get('start', 0), 3),
                "text": segment.get('text', '').strip(),
                "tokens": segment.get('tokens', []),
                "temperature": None,
                "avg_logprob": round(segment.get('avg_logprob', 0), 6),
                "compression_ratio": round(segment.get('compression_ratio', 0), 3),
                "no_speech_prob": round(segment.get('no_speech_prob', 0), 6)
            }
            
            # Add word-level timestamps if available
            if segment.get('words') and kwargs.get('word_timestamps', False):
                segment_data['words'] = []
                for word in segment['words']:
                    word_data = {
                        "word": word.get('word', ''),
                        "start": round(word.get('start', 0), 3),
                        "end": round(word.get('end', 0), 3),
                        "probability": round(word.get('probability', 0), 3)
                    }
                    segment_data['words'].append(word_data)
                    word_level_data.append(word_data)
            
            segments_list.append(segment_data)
            full_text_parts.append(segment.get('text', '').strip())
            
            # Track duration
            if segment.get('end', 0) > duration_seconds:
                duration_seconds = segment.get('end', 0)
            
            # Show progress
            if i < 10:  # Show first 10 segments
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()[:80]
                print(f"Segment {i+1}: [{start:.1f}s-{end:.1f}s] {text}...")
            elif i == 10:
                print("... (showing first 10 segments)")
    
    # Extract language info from result
    if 'language' in result:
        detected_language = result['language']
    
    # Create comprehensive JSON structure
    json_data = {
        "metadata": {
            # File information
            "file": os.path.basename(audio_path),
            "file_path": os.path.abspath(audio_path),
            "file_size_bytes": os.path.getsize(audio_path),
            
            # Model information
            "model": model_name,
            "device": device,
            "compute_type": "float32",  # mlx-whisper default
            "cpu_threads": None,
            "num_workers": 1,
            
            # Detection results
            "language": detected_language,
            "language_probability": result.get('language_probability', 0.0),
            "duration_seconds": round(duration_seconds, 3),
            "_audio_features": None,
            
            # Processing information
            "processing_time_seconds": round(processing_time, 3),
            "processing_speed": round(duration_seconds / processing_time, 2) if processing_time > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # VAD information (not directly available in mlx-whisper)
            "vad_enabled": False,
            "vad_parameters": None
        },
        
        "parameters": {
            # All parameters used for transcription
            "language": kwargs.get('language'),
            "task": kwargs.get('task', 'transcribe'),
            "beam_size": kwargs.get('beam_size', 5),
            "best_of": kwargs.get('best_of', 5),
            "patience": kwargs.get('patience', 1.0),
            "temperature": kwargs.get('temperature', 0.0),
            "length_penalty": kwargs.get('length_penalty', 1.0),
            "repetition_penalty": kwargs.get('repetition_penalty', 1.0),
            "no_repeat_ngram_size": kwargs.get('no_repeat_ngram_size', 0),
            "compression_ratio_threshold": kwargs.get('compression_ratio_threshold', 2.4),
            "log_prob_threshold": kwargs.get('log_prob_threshold', -1.0),
            "no_speech_threshold": kwargs.get('no_speech_threshold', 0.6),
            "condition_on_previous_text": kwargs.get('condition_on_previous_text', True),
            "initial_prompt": kwargs.get('initial_prompt'),
            "prefix": kwargs.get('prefix'),
            "suppress_blank": kwargs.get('suppress_blank', True),
            "suppress_tokens": kwargs.get('suppress_tokens', [-1]),
            "without_timestamps": kwargs.get('without_timestamps', False),
            "max_initial_timestamp": kwargs.get('max_initial_timestamp', 1.0),
            "word_timestamps": kwargs.get('word_timestamps', False),
            "prepend_punctuations": kwargs.get('prepend_punctuations',  '"\'¿([{-'),
            "append_punctuations": kwargs.get('append_punctuations', '"\'.。,!?:)}'),
            "vad_filter": False,
            "vad_parameters": None,
            "model": model_name
        },
        "transcription": {
            "full_text": " ".join(full_text_parts),
            "word_count": len(" ".join(full_text_parts).split()),
            "segment_count": len(segments_list),
            "character_count": len(" ".join(full_text_parts)),
            "average_segment_duration": round(sum(s['duration'] for s in segments_list) / len(segments_list), 3) if segments_list else 0,
            "word_level_timestamps_available": bool(word_level_data)
        },
        
        "segments": segments_list,
        
        "words": word_level_data if kwargs.get('word_timestamps', False) else None,
        
        "statistics": {
            "total_segments": len(segments_list),
            "total_words": len(word_level_data) if word_level_data else None,
            "average_words_per_segment": round(len(word_level_data) / len(segments_list), 2) if word_level_data and segments_list else None,
            "speech_rate_wpm": round((len(word_level_data) / duration_seconds) * 60, 2) if word_level_data and duration_seconds > 0 else None,
            "silence_detected": any(s['no_speech_prob'] > 0.5 for s in segments_list),
            "average_confidence": round(sum(s['avg_logprob'] for s in segments_list) / len(segments_list), 6) if segments_list else 0
        }
    }
    
    print(f"\n--- Summary ---")
    print(f"Language: {detected_language}")
    print(f"Duration: {duration_seconds:.3f} seconds")
    print(f"Processing time: {processing_time:.3f} seconds")
    if processing_time > 0:
        print(f"Speed: {duration_seconds/processing_time:.2f}x real-time")
    print(f"Segments: {len(segments_list)}")
    print(f"Words: {len(word_level_data) if word_level_data else 'Not extracted'}")
    print(f"Average confidence: {json_data['statistics']['average_confidence']:.6f}")
    
    return json_data


def save_json(data, audio_path, suffix="_transcription"):
    """Save transcription data to JSON file"""
    if not data:
        return None
    
    # Create output filename
    base_name = os.path.splitext(audio_path)[0]
    json_path = f"{base_name}{suffix}.json"
    
    # Save JSON with pretty formatting
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON saved to: {json_path}")
    print(f"File size: {os.path.getsize(json_path)} bytes")
    return json_path


def setup_argparse():
    """Setup command line argument parser - only audio file and config file"""
    parser = argparse.ArgumentParser(description="Advanced Whisper transcription with mlx-whisper and JSON configuration")
    
    # Only two required arguments
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--config", "-c", required=True, help="Path to JSON configuration file")
    
    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    print("=== Advanced MLX-Whisper Transcription with JSON Config ===")
    print(f"Audio file: {args.audio_file}")
    print(f"Config file: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        sys.exit(1)
    
    # Use configuration as-is (no command line overrides)
    final_params = config
    
    print(f"\n✓ Configuration loaded:")
    print(f"  Model: {final_params.get('model', 'base')}")
    print(f"  Device: {final_params.get('device', 'gpu')}")
    print(f"  Language: {final_params.get('language', 'auto-detect')}")
    print(f"  Word timestamps: {final_params.get('word_timestamps', False)}")
    
    # Transcribe
    result = transcribe_advanced(args.audio_file, **final_params)
    
    # Save to JSON
    if result:
        json_path = save_json(result, args.audio_file)
        
        # Show JSON structure preview
        print(f"\n--- JSON Structure ---")
        print("Main keys:", list(result.keys()))
        print("Parameters configured:", len([k for k, v in result["parameters"].items() if v is not None]))
        print("Word-level data:", "Available" if result["words"] else "Not extracted")
