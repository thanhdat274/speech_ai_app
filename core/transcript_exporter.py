"""
Transcript Exporter - Export transcripts to various formats
=============================================================================
"""

import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path


class TranscriptExporter:
    """Export transcripts to SRT, VTT, JSON, TXT formats."""

    def __init__(self, transcript_buffer):
        self._buffer = transcript_buffer

    def export_srt(self, output_path: str) -> str:
        """Export to SRT subtitle format."""
        segments = self._buffer.get_all_segments()

        srt_content = ""
        for idx, seg in enumerate(segments, 1):
            start = self._format_timestamp(seg['start'])
            end = self._format_timestamp(seg['end'])
            text = seg['text']

            srt_content += f"{idx}\n"
            srt_content += f"{start} --> {end}\n"
            srt_content += f"{text}\n\n"

        Path(output_path).write_text(srt_content, encoding='utf-8')
        return output_path

    def export_vtt(self, output_path: str) -> str:
        """Export to VTT subtitle format."""
        segments = self._buffer.get_all_segments()

        vtt_content = "WEBVTT\n\n"
        for idx, seg in enumerate(segments, 1):
            start = self._format_vtt_timestamp(seg['start'])
            end = self._format_vtt_timestamp(seg['end'])
            text = seg['text']

            vtt_content += f"{idx}\n"
            vtt_content += f"{start} --> {end}\n"
            vtt_content += f"{text}\n\n"

        Path(output_path).write_text(vtt_content, encoding='utf-8')
        return output_path

    def export_json(self, output_path: str) -> str:
        """Export to JSON with metadata."""
        segments = self._buffer.get_all_segments()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_segments": len(segments),
            "total_duration_s": (segments[-1]['end'] - segments[0]['start']
                                if segments else 0),
            "segments": segments
        }

        Path(output_path).write_text(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        return output_path

    def export_txt(self, output_path: str) -> str:
        """Export plain text (no timestamps)."""
        segments = self._buffer.get_all_segments()
        text = "\n\n".join(seg['text'] for seg in segments)

        Path(output_path).write_text(text, encoding='utf-8')
        return output_path

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to SRT timestamp: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

    @staticmethod
    def _format_vtt_timestamp(seconds: float) -> str:
        """Format seconds to VTT timestamp: HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
