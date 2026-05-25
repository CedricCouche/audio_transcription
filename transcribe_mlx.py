#!/usr/bin/env python3
"""
Audio transcription using mlx-whisper with JSON configuration.
Output: JSON file with segments and metadata.
"""

import mlx_whisper
import json
import sys
import os
import time
import argparse

KNOWN_PARAMS = {
    # Model selection
    'model',
    # Direct transcribe() params
    'verbose', 'temperature', 'compression_ratio_threshold', 'logprob_threshold',
    'no_speech_threshold', 'condition_on_previous_text', 'initial_prompt',
    'word_timestamps', 'prepend_punctuations', 'append_punctuations',
    'clip_timestamps', 'hallucination_silence_threshold',
    # decode_options passed via **kwargs
    'language', 'task', 'beam_size', 'best_of', 'patience', 'length_penalty',
    'suppress_blank', 'suppress_tokens', 'without_timestamps', 'max_initial_timestamp',
    'fp16', 'prefix',
    # Accepted as compat alias for logprob_threshold
    'log_prob_threshold',
}

MODEL_MAP = {
    'tiny':           'mlx-community/whisper-tiny-mlx',
    'base':           'mlx-community/whisper-base-mlx',
    'small':          'mlx-community/whisper-small-mlx',
    'medium':         'mlx-community/whisper-medium-mlx',
    'large':          'mlx-community/whisper-large-v3-mlx',
    'large-v2':       'mlx-community/whisper-large-v2-mlx',
    'large-v3':       'mlx-community/whisper-large-v3-mlx',
    'large-v3-turbo': 'mlx-community/whisper-large-v3-turbo',
}


def resolve_model(model):
    return MODEL_MAP.get(model, model)


def load_config(config_path):
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


def transcribe_mlx(audio_path, **kwargs):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return None

    path_or_hf_repo = resolve_model(kwargs.get('model', 'mlx-community/whisper-large-v3-mlx'))

    # Accept log_prob_threshold as compat alias
    logprob_threshold = kwargs.get('logprob_threshold', kwargs.get('log_prob_threshold', -1.0))

    # temperature can be a float or a list; mlx-whisper accepts both
    temperature = kwargs.get('temperature', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    if isinstance(temperature, list):
        temperature = tuple(temperature)

    # beam_size, best_of, and patience are not yet supported by mlx-whisper
    decode_options = {k: v for k, v in {
        'language':             kwargs.get('language'),
        'task':                 kwargs.get('task', 'transcribe'),
        'length_penalty':       kwargs.get('length_penalty'),
        'suppress_blank':       kwargs.get('suppress_blank', True),
        'suppress_tokens':      kwargs.get('suppress_tokens', '-1'),
        'without_timestamps':   kwargs.get('without_timestamps', False),
        'max_initial_timestamp': kwargs.get('max_initial_timestamp', 1.0),
        'fp16':                 kwargs.get('fp16', True),
        'prefix':               kwargs.get('prefix'),
    }.items() if v is not None}

    print(f"Model: {path_or_hf_repo}")
    print(f"Transcribing: {audio_path}")

    start_time = time.time()

    try:
        result = mlx_whisper.transcribe(
            audio=audio_path,
            path_or_hf_repo=path_or_hf_repo,
            temperature=temperature,
            compression_ratio_threshold=kwargs.get('compression_ratio_threshold', 2.4),
            logprob_threshold=logprob_threshold,
            no_speech_threshold=kwargs.get('no_speech_threshold', 0.6),
            condition_on_previous_text=kwargs.get('condition_on_previous_text', True),
            initial_prompt=kwargs.get('initial_prompt'),
            word_timestamps=kwargs.get('word_timestamps', False),
            prepend_punctuations=kwargs.get('prepend_punctuations', '"\'¿([{-'),
            append_punctuations=kwargs.get('append_punctuations', '"\'.。,!?:)]}'),
            clip_timestamps=kwargs.get('clip_timestamps', '0'),
            hallucination_silence_threshold=kwargs.get('hallucination_silence_threshold'),
            **decode_options,
        )
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

    processing_time = time.time() - start_time

    segments_list = []
    full_text_parts = []
    word_level_data = []
    duration_seconds = 0.0

    print("\n--- Processing segments ---")
    for i, segment in enumerate(result.get('segments', [])):
        segment_data = {
            "id": i,
            "seek": segment.get('seek', 0),
            "start": round(segment.get('start', 0), 3),
            "end": round(segment.get('end', 0), 3),
            "duration": round(segment.get('end', 0) - segment.get('start', 0), 3),
            "text": segment.get('text', '').strip(),
            "tokens": segment.get('tokens', []),
            "temperature": segment.get('temperature'),
            "avg_logprob": round(segment.get('avg_logprob', 0), 6),
            "compression_ratio": round(segment.get('compression_ratio', 0), 3),
            "no_speech_prob": round(segment.get('no_speech_prob', 0), 6),
        }

        if segment.get('words') and kwargs.get('word_timestamps', False):
            segment_data['words'] = []
            for word in segment['words']:
                word_data = {
                    "word": word.get('word', ''),
                    "start": round(word.get('start', 0), 3),
                    "end": round(word.get('end', 0), 3),
                    "probability": round(word.get('probability', 0), 3),
                }
                segment_data['words'].append(word_data)
                word_level_data.append(word_data)

        segments_list.append(segment_data)
        full_text_parts.append(segment.get('text', '').strip())

        if segment.get('end', 0) > duration_seconds:
            duration_seconds = segment.get('end', 0)

        if i < 10:
            print(f"Segment {i+1}: [{segment.get('start', 0):.1f}s-{segment.get('end', 0):.1f}s] {segment.get('text', '').strip()[:80]}...")
        elif i == 10:
            print("... (showing first 10 segments)")

    detected_language = result.get('language', kwargs.get('language', 'unknown'))
    full_text = " ".join(full_text_parts)

    json_data = {
        "metadata": {
            "file": os.path.basename(audio_path),
            "file_path": os.path.abspath(audio_path),
            "file_size_bytes": os.path.getsize(audio_path),
            "model": path_or_hf_repo,
            "device": "mlx",
            "compute_type": "float16" if kwargs.get('fp16', True) else "float32",
            "cpu_threads": None,
            "num_workers": None,
            "language": detected_language,
            "language_probability": 0.0,
            "duration_seconds": round(duration_seconds, 3),
            "_audio_features": None,
            "processing_time_seconds": round(processing_time, 3),
            "processing_speed": round(duration_seconds / processing_time, 2) if processing_time > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vad_enabled": False,
            "vad_parameters": None,
        },
        "parameters": {
            "language": kwargs.get('language'),
            "task": kwargs.get('task', 'transcribe'),
            "beam_size": kwargs.get('beam_size'),
            "best_of": kwargs.get('best_of'),
            "patience": kwargs.get('patience', 1.0),
            "temperature": list(temperature) if isinstance(temperature, tuple) else temperature,
            "length_penalty": kwargs.get('length_penalty', 1.0),
            "compression_ratio_threshold": kwargs.get('compression_ratio_threshold', 2.4),
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": kwargs.get('no_speech_threshold', 0.6),
            "condition_on_previous_text": kwargs.get('condition_on_previous_text', True),
            "initial_prompt": kwargs.get('initial_prompt'),
            "word_timestamps": kwargs.get('word_timestamps', False),
            "prepend_punctuations": kwargs.get('prepend_punctuations', '"\'¿([{-'),
            "append_punctuations": kwargs.get('append_punctuations', '"\'.。,!?:)]}'),
            "fp16": kwargs.get('fp16', True),
            "clip_timestamps": kwargs.get('clip_timestamps', '0'),
            "hallucination_silence_threshold": kwargs.get('hallucination_silence_threshold'),
            "model": path_or_hf_repo,
        },
        "transcription": {
            "full_text": full_text,
            "word_count": len(full_text.split()),
            "segment_count": len(segments_list),
            "character_count": len(full_text),
            "average_segment_duration": round(sum(s['duration'] for s in segments_list) / len(segments_list), 3) if segments_list else 0,
            "word_level_timestamps_available": bool(word_level_data),
        },
        "segments": segments_list,
        "words": word_level_data if kwargs.get('word_timestamps', False) else None,
        "statistics": {
            "total_segments": len(segments_list),
            "total_words": len(word_level_data) if word_level_data else None,
            "average_words_per_segment": round(len(word_level_data) / len(segments_list), 2) if word_level_data and segments_list else None,
            "speech_rate_wpm": round((len(word_level_data) / duration_seconds) * 60, 2) if word_level_data and duration_seconds > 0 else None,
            "silence_detected": any(s['no_speech_prob'] > 0.5 for s in segments_list),
            "average_confidence": round(sum(s['avg_logprob'] for s in segments_list) / len(segments_list), 6) if segments_list else 0,
        },
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
    if not data:
        return None
    base_name = os.path.splitext(audio_path)[0]
    json_path = f"{base_name}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to: {json_path}")
    print(f"File size: {os.path.getsize(json_path)} bytes")
    return json_path


def setup_argparse():
    parser = argparse.ArgumentParser(description="MLX-Whisper transcription with JSON configuration")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--config", "-c", required=True, help="Path to JSON configuration file")
    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    print("=== MLX-Whisper Transcription with JSON Config ===")
    print(f"Audio file: {args.audio_file}")
    print(f"Config file: {args.config}")

    config = load_config(args.config)
    if not config:
        sys.exit(1)

    final_params = config

    print(f"\n✓ Configuration loaded:")
    print(f"  Model: {resolve_model(final_params.get('model', 'mlx-community/whisper-large-v3-mlx'))}")
    print(f"  Language: {final_params.get('language') or 'auto-detect'}")
    print(f"  Word timestamps: {final_params.get('word_timestamps', False)}")

    result = transcribe_mlx(args.audio_file, **final_params)

    if result:
        json_path = save_json(result, args.audio_file)
        print(f"\n--- JSON Structure ---")
        print("Main keys:", list(result.keys()))
        print("Parameters configured:", len([k for k, v in result["parameters"].items() if v is not None]))
        print("Word-level data:", "Available" if result["words"] else "Not extracted")
    else:
        print("Transcription failed.")

# python transcribe_mlx.py data/audio.mp3 --config configs/mlx_whisper.json
