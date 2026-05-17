#!/usr/bin/env python3
"""
Audio Transcription Formatter
Converts JSON transcription files to readable text with intelligent paragraph grouping.
"""

import json
import re
import argparse
from typing import List, Dict, Any
from datetime import timedelta


class TranscriptionFormatter:
    def __init__(self, pause_threshold: float = 3.0, min_paragraph_length: int = 200):
        """
        Initialize the formatter with configurable parameters.

        Args:
            pause_threshold: Minimum pause duration (seconds) to create new paragraph
            min_paragraph_length: Minimum characters in a paragraph before considering breaks
        """
        self.pause_threshold = pause_threshold
        self.min_paragraph_length = min_paragraph_length

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to readable timestamp format (MM:SS)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def detect_natural_breaks(self, text: str) -> bool:
        """
        Detect natural paragraph breaks based on content patterns.

        Args:
            text: Text to analyze

        Returns:
            True if text suggests a natural paragraph break
        """
        break_patterns = [
            r'\b(okay|alright|so|now|next|moving on|let\'s|well)\b',
            r'\b(definition|example|theorem|proof|question)\b',
            r'[.!?]\s*[A-Z]',
            r'\b(first|second|third|finally|in conclusion)\b',
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in break_patterns)

    def group_into_paragraphs(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group segments into paragraphs based on timestamps and content analysis.

        Args:
            segments: List of transcription segments

        Returns:
            List of paragraph dictionaries with combined text and metadata
        """
        if not segments:
            return []

        paragraphs = []
        current_paragraph = {
            'start_time': segments[0]['start'],
            'end_time': segments[0]['end'],
            'text': segments[0]['text'].strip(),
            'segment_count': 1
        }

        for i in range(1, len(segments)):
            current_segment = segments[i]
            prev_segment = segments[i - 1]

            pause_duration = current_segment['start'] - prev_segment['end']

            should_break = (
                pause_duration >= self.pause_threshold and
                len(current_paragraph['text']) >= self.min_paragraph_length
            ) or (
                len(current_paragraph['text']) >= self.min_paragraph_length * 2 and
                self.detect_natural_breaks(current_segment['text'])
            )

            if should_break:
                paragraphs.append(current_paragraph)
                current_paragraph = {
                    'start_time': current_segment['start'],
                    'end_time': current_segment['end'],
                    'text': current_segment['text'].strip(),
                    'segment_count': 1
                }
            else:
                current_paragraph['text'] += ' ' + current_segment['text'].strip()
                current_paragraph['end_time'] = current_segment['end']
                current_paragraph['segment_count'] += 1

        if current_paragraph['text']:
            paragraphs.append(current_paragraph)

        return paragraphs

    def clean_text(self, text: str) -> str:
        """Clean and format text for better readability."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        return text.strip()

    def format_sentences_on_newlines(self, text: str) -> str:
        """
        Format text so each sentence starts on a new line within paragraphs.
        Requires ≥3 word characters before the punctuation to avoid splitting
        on abbreviations like "Dr." or "U.S.".
        """
        parts = re.split(r'(?<=\w{3}[.!?])\s+(?=[A-Z])', text)
        return '\n'.join(p.strip() for p in parts if p.strip())

    def format_transcription(self, json_data: Dict[str, Any],
                             include_timestamps: bool = True,
                             include_metadata: bool = True) -> str:
        """
        Format the entire transcription into readable text.

        Args:
            json_data: Parsed JSON transcription data
            include_timestamps: Whether to include timestamp markers
            include_metadata: Whether to include file metadata at the top

        Returns:
            Formatted text string
        """
        output_lines = []

        if include_metadata and 'metadata' in json_data:
            metadata = json_data['metadata']
            output_lines.extend([
                "=" * 60,
                "TRANSCRIPTION METADATA",
                "=" * 60,
                f"File: {metadata.get('file', 'Unknown')}",
                f"Duration: {self.format_timestamp(metadata.get('duration_seconds', 0))}",
                f"Language: {metadata.get('language', 'Unknown')}",
                f"Word Count: {json_data.get('transcription', {}).get('word_count', 'Unknown')}",
                f"Processing Date: {metadata.get('timestamp', 'Unknown')}",
                "=" * 60,
                ""
            ])

        segments = json_data.get('segments', [])
        paragraphs = self.group_into_paragraphs(segments)

        for paragraph in paragraphs:
            if include_timestamps:
                start_time = self.format_timestamp(paragraph['start_time'])
                end_time = self.format_timestamp(paragraph['end_time'])
                output_lines.append(f"[{start_time} - {end_time}]")

            clean_text = self.clean_text(paragraph['text'])
            formatted_text = self.format_sentences_on_newlines(clean_text)
            output_lines.append(formatted_text)
            output_lines.append("")

        return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description="Format a JSON transcription into readable text")
    parser.add_argument(
        "--input", "-i",
        default="data/audio_transcription.json",
        help="Path to transcription JSON file (default: data/audio_transcription.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/formatted_transcription.txt",
        help="Path for the formatted output file (default: data/formatted_transcription.txt)"
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=3.0,
        help="Pause duration in seconds that triggers a new paragraph (default: 3.0)"
    )
    parser.add_argument(
        "--min-paragraph-length",
        type=int,
        default=200,
        help="Minimum characters before a paragraph break is considered (default: 200)"
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Omit timestamp markers from output"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Omit metadata header from output"
    )
    args = parser.parse_args()

    formatter = TranscriptionFormatter(
        pause_threshold=args.pause_threshold,
        min_paragraph_length=args.min_paragraph_length,
    )

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        formatted_text = formatter.format_transcription(
            json_data,
            include_timestamps=not args.no_timestamps,
            include_metadata=not args.no_metadata,
        )

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        print(f"✅ Transcription formatted successfully!")
        print(f"📄 Output saved to: {args.output}")

        segments = json_data.get('segments', [])
        paragraphs = formatter.group_into_paragraphs(segments)
        print(f"📊 Statistics:")
        print(f"   - Total segments: {len(segments)}")
        print(f"   - Formatted paragraphs: {len(paragraphs)}")

        print(f"\n📖 Preview (first 500 characters):")
        print("-" * 50)
        print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)

    except FileNotFoundError:
        print(f"❌ Error: {args.input} not found")
        print("Please ensure the JSON transcription file exists before running the formatter.")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
