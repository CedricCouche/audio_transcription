#!/usr/bin/env python3
"""
Audio Transcription Formatter
Converts JSON transcription files to readable text with intelligent paragraph grouping.
"""

import json
import re
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
        # Patterns that suggest topic transitions or natural breaks
        break_patterns = [
            r'\b(okay|alright|so|now|next|moving on|let\'s|well)\b',  # Transition words
            r'\b(definition|example|theorem|proof|question)\b',        # Academic markers
            r'[.!?]\s*[A-Z]',  # Sentence endings followed by capital letters
            r'\b(first|second|third|finally|in conclusion)\b',        # Enumeration
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
            prev_segment = segments[i-1]
            
            # Calculate pause between segments
            pause_duration = current_segment['start'] - prev_segment['end']
            
            # Check if we should start a new paragraph
            should_break = (
                pause_duration >= self.pause_threshold and 
                len(current_paragraph['text']) >= self.min_paragraph_length
            ) or (
                len(current_paragraph['text']) >= self.min_paragraph_length * 2 and
                self.detect_natural_breaks(current_segment['text'])
            )
            
            if should_break:
                # Finalize current paragraph
                paragraphs.append(current_paragraph)
                
                # Start new paragraph
                current_paragraph = {
                    'start_time': current_segment['start'],
                    'end_time': current_segment['end'],
                    'text': current_segment['text'].strip(),
                    'segment_count': 1
                }
            else:
                # Add to current paragraph
                current_paragraph['text'] += ' ' + current_segment['text'].strip()
                current_paragraph['end_time'] = current_segment['end']
                current_paragraph['segment_count'] += 1
        
        # Don't forget the last paragraph
        if current_paragraph['text']:
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def clean_text(self, text: str) -> str:
        """Clean and format text for better readability."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common transcription issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence end
        
        # Capitalize first letter of sentences
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text.strip()
    
    def format_sentences_on_newlines(self, text: str) -> str:
        """
        Format text so each sentence starts on a new line within paragraphs.
        
        Args:
            text: Input text to format
            
        Returns:
            Text with sentences separated by newlines
        """
        # Split on sentence endings followed by space and capital letter
        # This regex looks for: period/exclamation/question mark + space + capital letter
        sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)
        
        # Rejoin sentences with their punctuation and add newlines
        formatted_lines = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
                # This is a sentence with punctuation
                sentence = sentences[i] + sentences[i + 1]
                formatted_lines.append(sentence.strip())
                i += 2
            else:
                # Last sentence or sentence without clear ending
                if sentences[i].strip():
                    formatted_lines.append(sentences[i].strip())
                i += 1
        
        # Join with newlines, but avoid empty lines
        return '\n'.join(line for line in formatted_lines if line.strip())
    
    
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
        
        # Add metadata header if requested
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
        
        # Process segments into paragraphs
        segments = json_data.get('segments', [])
        paragraphs = self.group_into_paragraphs(segments)
        
        # Format paragraphs
        for i, paragraph in enumerate(paragraphs, 1):
            if include_timestamps:
                start_time = self.format_timestamp(paragraph['start_time'])
                end_time = self.format_timestamp(paragraph['end_time'])
                output_lines.append(f"[{start_time} - {end_time}]")
            
            # Clean and format the text
            clean_text = self.clean_text(paragraph['text'])
            formatted_text = self.format_sentences_on_newlines(clean_text)
            output_lines.append(formatted_text)
            output_lines.append("")  # Empty line between paragraphs
        
        return "\n".join(output_lines)

def main():
    """Example usage of the TranscriptionFormatter."""
    # Configuration options
    formatter = TranscriptionFormatter(
        pause_threshold=1.0,      # 3 seconds pause for new paragraph
        min_paragraph_length=500  # Minimum 200 characters per paragraph
    )
    
    # Load JSON file
    try:
        with open('data/audio_transcription.json', 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # Format the transcription
        formatted_text = formatter.format_transcription(
            json_data,
            include_timestamps=True,
            include_metadata=True
        )
        
        # Save to file
        output_filename = 'data/formatted_transcription.txt'
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(formatted_text)
        
        print(f"✅ Transcription formatted successfully!")
        print(f"📄 Output saved to: {output_filename}")
        print(f"📊 Statistics:")
        print(f"   - Total segments: {len(json_data.get('segments', []))}")
        
        # Count paragraphs
        paragraphs = formatter.group_into_paragraphs(json_data.get('segments', []))
        print(f"   - Formatted paragraphs: {len(paragraphs)}")
        
        # Show preview
        print(f"\n📖 Preview (first 500 characters):")
        print("-" * 50)
        print(formatted_text[:500] + "..." if len(formatted_text) > 500 else formatted_text)
        
    except FileNotFoundError:
        print("❌ Error: transcription_short.json not found")
        print("Please ensure the JSON file is in the same directory as this script")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()