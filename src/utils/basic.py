def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

audio_file_extensions = [
    '.mp3',  # MPEG-1 Audio Layer 3 (most common lossy format)
    '.wav',  # Waveform Audio File Format (uncompressed audio)
    '.flac',  # Free Lossless Audio Codec (lossless compression)
]

video_file_extensions = [
    ".mp4",  # Widely used, supported across most devices and platforms
    ".mkv",  # High-quality videos, supports multiple audio and subtitle tracks
    ".mov",  # Common for Apple devices and high-quality media
]

SUPPORTED_FILES = ' '.join(video_file_extensions + audio_file_extensions + ['.srt'])