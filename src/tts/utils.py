import re
from datetime import timedelta

import numpy as np
import srt


def remove_tags(text):
    CLEAN1 = re.compile('<.*?>')
    CLEAN2 = re.compile('{.*?}')
    CLEAN3 = re.compile('\[.*?\]')
    text = re.sub(CLEAN1, '', text)
    text = re.sub(CLEAN2, '', text)
    text = re.sub(CLEAN3, '', text)
    return text


def read_srt(file):
    with open(file) as f:
        subs = f.read()
    return subs


def normalize(value, old_min, old_max, new_min, new_max):
    # Ensure the old range is not zero to avoid division by zero
    if old_max - old_min == 0:
        raise ValueError("Old range cannot be zero.")

    # Apply the normalization formula
    normalized_value = new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    return normalized_value


def check_if_extra_time_is_needed(text, segment_duration, words_per_second=170 / 60):
    max_words = words_per_second * segment_duration
    # words = re.split('\s+', text)
    words = re.findall(r"\b\w+\b", text)
    number_of_words = len(words)
    extra_seconds_needed = 0
    if number_of_words > max_words:
        extra_seconds_needed = (number_of_words - max_words) / words_per_second
    return extra_seconds_needed, number_of_words


def find_available_space(subtitles, index):
    end = subtitles[index].end
    start = subtitles[index].start
    if index == 0:
        previous_end = start
    else:
        previous_end = subtitles[index - 1].end
    if index == len(subtitles) - 1:
        next_start = end
    else:
        next_start = subtitles[index + 1].start
    left_space = (start - previous_end).total_seconds()
    right_space = (next_start - end).total_seconds()
    return left_space, right_space


def expand_segment_if_possible(sub, left_space, right_space, extra_seconds_needed):
    # Calculate the expansion on each side based on available space and needed extra time
    left_expand = min(left_space, extra_seconds_needed / 2)
    right_expand = min(right_space, extra_seconds_needed - left_expand)

    # Apply the calculated expansion to the segment's start and end times
    sub.start -= timedelta(seconds=left_expand)
    sub.end += timedelta(seconds=right_expand)


def generate_full_audio_with_silence(subtitles, audio_segments, sample_rate=24000):
    # Calculate the total duration of the output audio in seconds
    total_duration = subtitles[-1].end.total_seconds()
    total_samples = int(total_duration * sample_rate)

    # Initialize an array of silence for the total duration
    full_audio = np.zeros(total_samples, dtype=np.float32)

    for subtitle, audio_segment in zip(subtitles, audio_segments):
        # Determine the start and end positions (in samples) for the current subtitle audio
        start_sample = int(subtitle.start.total_seconds() * sample_rate)
        end_sample = min(start_sample + len(audio_segment), total_samples)

        # Place the audio segment at the calculated start position
        full_audio[start_sample:end_sample] = audio_segment[:end_sample - start_sample]

    return full_audio


def add_silence(array: list, sr: int, target_duration: float):
    number_of_frames = np.array(array).size
    target_frames = target_duration * sr
    silence_frames = int(target_frames - number_of_frames)
    silence = np.zeros(silence_frames)
    left_silence, right_silence = np.array_split(silence, 2)
    array = np.concatenate([left_silence, array, right_silence])
    return array


def get_longest_srt_segment(file_path):
    """
    Get the timestamps (in seconds) of the longest segment in an SRT file.

    Parameters:
        file_path: The path to the SRT file.

    Returns:
        A tuple containing (start_time_in_seconds, end_time_in_seconds, duration_in_seconds).
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the SRT file
    subtitles = list(srt.parse(content))

    longest_segment = None
    max_duration = 0.0

    for subtitle in subtitles:
        start_time = subtitle.start.total_seconds()
        end_time = subtitle.end.total_seconds()
        duration = end_time - start_time

        if duration > max_duration:
            max_duration = duration
            longest_segment = (start_time, end_time, duration)

    return longest_segment

def add_segment(subs, text, start, end):
    subs = list(srt.parse(subs))
    sub = srt.Subtitle(index=len(subs) + 1, start=timedelta(seconds=float(start)), end=timedelta(seconds=float(end)),
                       content=text)
    subs.append(sub)

    return srt.compose(subs), '', float(end), float(end) + 1.0

def remove_last(subs):
    subs = list(srt.parse(subs))
    if len(subs) > 0:
        subs.pop()
    return srt.compose(subs)