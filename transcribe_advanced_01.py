#!/usr/bin/env python3
"""
Advanced audio transcription using faster-whisper with all configurable parameters
Output: JSON file with segments and metadata
"""

from faster_whisper import WhisperModel
import json
import sys
import os
import time
import argparse

def transcribe_advanced(audio_path, **kwargs):
    """
    Advanced transcribe with all possible parameters
    
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
    model_size = kwargs.get('model_size', 'medium')
    device = kwargs.get('device', 'cuda')
    compute_type = kwargs.get('compute_type', 'int8')
    cpu_threads = kwargs.get('cpu_threads', 0)  # 0 = use all available
    num_workers = kwargs.get('num_workers', 1)
    download_root = kwargs.get('download_root', None)
    local_files_only = kwargs.get('local_files_only', False)
    
    print(f"Loading Whisper model: {model_size}")
    print(f"Device: {device}, Compute type: {compute_type}")
    
    # Initialize the model with all parameters
    model = WhisperModel(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
        local_files_only=local_files_only
    )
    
    print(f"Transcribing: {audio_path}")
    print(f"Parameters: {json.dumps({k: v for k, v in kwargs.items() if k.startswith(('beam_', 'temperature', 'patience', 'length_', 'language', 'task', 'vad_', 'word_', 'prepend_', 'suppress_'))}, indent=2)}")
    
    start_time = time.time()
    
    # Transcribe with all possible parameters
    segments, info = model.transcribe(
        audio=audio_path,
        
        # Language and task parameters
        language=kwargs.get('language', None),  # None for auto-detection
        task=kwargs.get('task', 'transcribe'),  # 'transcribe' or 'translate'
        
        # Beam search parameters
        beam_size=kwargs.get('beam_size', 5),
        best_of=kwargs.get('best_of', 5),
        patience=kwargs.get('patience', 1.0),
        
        # Temperature parameters (for sampling)
        temperature=kwargs.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        
        # Length penalties
        length_penalty=kwargs.get('length_penalty', 1.0),
        repetition_penalty=kwargs.get('repetition_penalty', 1.0),
        no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 0),
        
        # Compression ratio threshold
        compression_ratio_threshold=kwargs.get('compression_ratio_threshold', 2.4),
        log_prob_threshold=kwargs.get('log_prob_threshold', -1.0),
        no_speech_threshold=kwargs.get('no_speech_threshold', 0.6),
        
        # Conditioning parameters
        condition_on_previous_text=kwargs.get('condition_on_previous_text', True),
        prompt_reset_on_temperature=kwargs.get('prompt_reset_on_temperature', 0.5),
        initial_prompt=kwargs.get('initial_prompt', None),
        prefix=kwargs.get('prefix', None),
        suppress_blank=kwargs.get('suppress_blank', True),
        suppress_tokens=kwargs.get('suppress_tokens', [-1]),
        
        # Timestamp parameters
        without_timestamps=kwargs.get('without_timestamps', False),
        max_initial_timestamp=kwargs.get('max_initial_timestamp', 1.0),
        
        # Word-level timestamps
        word_timestamps=kwargs.get('word_timestamps', False),
        prepend_punctuations = kwargs.get('prepend_punctuations', '"\'¿([{-'),
        append_punctuations=kwargs.get('append_punctuations', '"\'.。,!?:)]}'),
        
        # VAD (Voice Activity Detection) parameters
        vad_filter=kwargs.get('vad_filter', True),
        vad_parameters=kwargs.get('vad_parameters', {
            'threshold': 0.5,
            'min_speech_duration_ms': 250,
            'max_speech_duration_s': float('inf'),
            'min_silence_duration_ms': 2000,
            'window_size_samples': 1024,
            'speech_pad_ms': 400
        }),
        
        # Chunking parameters
        max_new_tokens=kwargs.get('max_new_tokens', None),
        chunk_length=kwargs.get('chunk_length', 30),
        clip_timestamps=kwargs.get('clip_timestamps', [0.0]),
        hallucination_silence_threshold=kwargs.get('hallucination_silence_threshold', None)
    )
    
    # Collect segments
    segments_list = []
    full_text_parts = []
    word_level_data = []
    
    print("\n--- Processing segments ---")
    for i, segment in enumerate(segments):
        segment_data = {
            "id": i,
            "seek": getattr(segment, 'seek', 0),
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "duration": round(segment.end - segment.start, 3),
            "text": segment.text.strip(),
            "tokens": getattr(segment, 'tokens', []),
            "temperature": getattr(segment, 'temperature', None),
            "avg_logprob": round(getattr(segment, 'avg_logprob', 0), 6),
            "compression_ratio": round(getattr(segment, 'compression_ratio', 0), 3),
            "no_speech_prob": round(getattr(segment, 'no_speech_prob', 0), 6)
        }
        
        # Add word-level timestamps if available
        if hasattr(segment, 'words') and segment.words:
            segment_data['words'] = []
            for word in segment.words:
                word_data = {
                    "word": word.word,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "probability": round(word.probability, 3)
                }
                segment_data['words'].append(word_data)
                word_level_data.append(word_data)
        
        segments_list.append(segment_data)
        full_text_parts.append(segment.text.strip())
        
        # Show progress
        if i < 10:  # Show first 10 segments
            print(f"Segment {i+1}: [{segment.start:.1f}s-{segment.end:.1f}s] {segment.text.strip()[:80]}...")
        elif i == 10:
            print("... (showing first 10 segments)")
    
    processing_time = time.time() - start_time
    
    # Create comprehensive JSON structure
    json_data = {
        "metadata": {
            # File information
            "file": os.path.basename(audio_path),
            "file_path": os.path.abspath(audio_path),
            "file_size_bytes": os.path.getsize(audio_path),
            
            # Model information
            "model": model_size,
            "device": device,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
            
            # Detection results
            "language": info.language,
            "language_probability": round(info.language_probability, 6),
            "duration_seconds": round(info.duration, 3),
            "_audio_features": getattr(info, 'all_language_probs', None),
            
            # Processing information
            "processing_time_seconds": round(processing_time, 3),
            "processing_speed": round(info.duration / processing_time, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # VAD information
            "vad_enabled": kwargs.get('vad_filter', True),
            "vad_parameters": kwargs.get('vad_parameters', {}) if kwargs.get('vad_filter', True) else None
        },
        
        "parameters": {
            # All parameters used for transcription
            "language": kwargs.get('language'),
            "task": kwargs.get('task', 'transcribe'),
            "beam_size": kwargs.get('beam_size', 5),
            "best_of": kwargs.get('best_of', 5),
            "patience": kwargs.get('patience', 1.0),
            "temperature": kwargs.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            "length_penalty": kwargs.get('length_penalty', 1.0),
            "repetition_penalty": kwargs.get('repetition_penalty', 1.0),
            "no_repeat_ngram_size": kwargs.get('no_repeat_ngram_size', 0),
            "compression_ratio_threshold": kwargs.get('compression_ratio_threshold', 2.4),
            "log_prob_threshold": kwargs.get('log_prob_threshold', -1.0),
            "no_speech_threshold": kwargs.get('no_speech_threshold', 0.6),
            "condition_on_previous_text": kwargs.get('condition_on_previous_text', True),
            "prompt_reset_on_temperature": kwargs.get('prompt_reset_on_temperature', 0.5),
            "initial_prompt": kwargs.get('initial_prompt'),
            "prefix": kwargs.get('prefix'),
            "suppress_blank": kwargs.get('suppress_blank', True),
            "suppress_tokens": kwargs.get('suppress_tokens', [-1]),
            "without_timestamps": kwargs.get('without_timestamps', False),
            "max_initial_timestamp": kwargs.get('max_initial_timestamp', 1.0),
            "word_timestamps": kwargs.get('word_timestamps', False),
            "prepend_punctuations": kwargs.get('prepend_punctuations',  '"\'¿([{-'),
            "append_punctuations": kwargs.get('append_punctuations', '"\'.。,!?:)]}'),
            "vad_filter": kwargs.get('vad_filter', True),
            "vad_parameters": kwargs.get('vad_parameters', {}),
            "max_new_tokens": kwargs.get('max_new_tokens'),
            "chunk_length": kwargs.get('chunk_length', 30),
            "clip_timestamps": kwargs.get('clip_timestamps', [0.0]),
            "hallucination_silence_threshold": kwargs.get('hallucination_silence_threshold')
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
            "speech_rate_wpm": round((len(word_level_data) / info.duration) * 60, 2) if word_level_data else None,
            "silence_detected": any(s['no_speech_prob'] > 0.5 for s in segments_list),
            "average_confidence": round(sum(s['avg_logprob'] for s in segments_list) / len(segments_list), 6) if segments_list else 0
        }
    }
    
    print(f"\n--- Summary ---")
    print(f"Language: {info.language} (confidence: {info.language_probability:.6f})")
    print(f"Duration: {info.duration:.3f} seconds")
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Speed: {info.duration/processing_time:.2f}x real-time")
    print(f"Segments: {len(segments_list)}")
    print(f"Words: {len(word_level_data) if word_level_data else 'Not extracted'}")
    print(f"Average confidence: {json_data['statistics']['average_confidence']:.6f}")
    
    return json_data

def save_json(data, audio_path, suffix="_advanced_transcription"):
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
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description="Advanced Whisper transcription with all parameters")
    
    # Required argument
    parser.add_argument("audio_file", help="Path to audio file")
    
    # Model parameters
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"], help="Model size")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"], help="Device to use")
    parser.add_argument("--compute-type", default="int8", choices=["int8", "float16", "float32"], help="Compute precision")
    parser.add_argument("--cpu-threads", type=int, default=0, help="Number of CPU threads (0 = all)")
    
    # Transcription parameters
    parser.add_argument("--language", help="Force language (e.g., 'en', 'fr', 'es')")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task type")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--best-of", type=int, default=5, help="Number of candidates for best-of search")
    parser.add_argument("--patience", type=float, default=1.0, help="Patience for beam search")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--compression-ratio-threshold", type=float, default=2.4, help="Compression ratio threshold")
    parser.add_argument("--log-prob-threshold", type=float, default=-1.0, help="Log probability threshold")
    parser.add_argument("--no-speech-threshold", type=float, default=0.6, help="No speech probability threshold")
    
    # Timestamp parameters
    parser.add_argument("--word-timestamps", action="store_true", help="Extract word-level timestamps")
    parser.add_argument("--without-timestamps", action="store_true", help="Disable timestamps")
    parser.add_argument("--max-initial-timestamp", type=float, default=1.0, help="Maximum initial timestamp")
    
    # VAD parameters
    parser.add_argument("--no-vad", action="store_true", help="Disable voice activity detection")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="VAD threshold")
    parser.add_argument("--min-speech-duration", type=int, default=250, help="Minimum speech duration (ms)")
    parser.add_argument("--min-silence-duration", type=int, default=2000, help="Minimum silence duration (ms)")
    
    # Other parameters
    parser.add_argument("--initial-prompt", help="Initial prompt to guide transcription")
    parser.add_argument("--chunk-length", type=int, default=30, help="Chunk length in seconds")
    parser.add_argument("--condition-on-previous-text", action="store_false", help="Disable conditioning on previous text")
    
    return parser

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    
    print("=== Advanced Whisper Transcription ===")
    print(f"Audio file: {args.audio_file}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    
    # Build parameters dictionary
    params = {
        'model_size': args.model,
        'device': args.device,
        'compute_type': args.compute_type,
        'cpu_threads': args.cpu_threads,
        'language': args.language,
        'task': args.task,
        'beam_size': args.beam_size,
        'best_of': args.best_of,
        'patience': args.patience,
        'length_penalty': args.length_penalty,
        'repetition_penalty': args.repetition_penalty,
        'compression_ratio_threshold': args.compression_ratio_threshold,
        'log_prob_threshold': args.log_prob_threshold,
        'no_speech_threshold': args.no_speech_threshold,
        'word_timestamps': args.word_timestamps,
        'without_timestamps': args.without_timestamps,
        'max_initial_timestamp': args.max_initial_timestamp,
        'vad_filter': not args.no_vad,
        'initial_prompt': args.initial_prompt,
        'chunk_length': args.chunk_length,
        'condition_on_previous_text': args.condition_on_previous_text,
        'vad_parameters': {
            'threshold': args.vad_threshold,
            'min_speech_duration_ms': args.min_speech_duration,
            'min_silence_duration_ms': args.min_silence_duration
        } if not args.no_vad else {}
    }
    
    # Transcribe
    result = transcribe_advanced(args.audio_file, **params)
    
    # Save to JSON
    if result:
        json_path = save_json(result, args.audio_file)
        
        # Show JSON structure preview
        print(f"\n--- JSON Structure ---")
        print("Main keys:", list(result.keys()))
        print("Parameters configured:", len([k for k, v in result["parameters"].items() if v is not None]))
        print("Word-level data:", "Available" if result["words"] else "Not extracted")
    else:
        print("Transcription failed.")

# Example usage:
# Basic : python advanced_transcribe.py Audio_01.mp3
# High-quality with word timestamps : python advanced_transcribe.py Audio_01.mp3 --model large-v3 --word-timestamps --beam-size 10
# Fast processing: python advanced_transcribe.py Audio_01.mp3 --model small --beam-size 1 --no-vad
# French with custom prompt: python advanced_transcribe.py Audio_01.mp3 --language fr --initial-prompt "This is a technical lecture about"