import json
import os
import re
import sys
import zlib
from typing import Callable, List, Optional, TextIO

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
        seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s["words"])),
        segments[-1]["end"] if segments else None,
    )


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
            self, result: dict, audio_path: str, options: Optional[dict] = None, **kwargs
    ):
        audio_basename = os.path.basename(audio_path)
        audio_basename = os.path.splitext(audio_basename)[0]
        output_path = os.path.join(
            self.output_dir, audio_basename + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)
        return output_path

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for segment in result["segments"]:
            print(segment["text"].strip(), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
            self,
            result: dict,
            options: Optional[dict] = None,
            *,
            max_line_width: Optional[int] = None,
            max_line_count: Optional[int] = None,
            highlight_words: bool = False,
            max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            # the next subtitle to yield (a list of word timings with whitespace)
            subtitle: List[dict] = []
            last: float = get_start(result["segments"]) or 0.0
            for segment in result["segments"]:
                chunk_index = 0
                words_count = max_words_per_line
                while chunk_index < len(segment["words"]):
                    remaining_words = len(segment["words"]) - chunk_index
                    if max_words_per_line > len(segment["words"]) - chunk_index:
                        words_count = remaining_words
                    for i, original_timing in enumerate(
                            segment["words"][chunk_index: chunk_index + words_count]
                    ):
                        timing = original_timing.copy()
                        long_pause = (
                                not preserve_segments and timing["start"] - last > 3.0
                        )
                        has_room = line_len + len(timing["word"]) <= max_line_width
                        seg_break = i == 0 and len(subtitle) > 0 and preserve_segments
                        if (
                                line_len > 0
                                and has_room
                                and not long_pause
                                and not seg_break
                        ):
                            # line continuation
                            line_len += len(timing["word"])
                        else:
                            # new line
                            timing["word"] = timing["word"].strip()
                            if (
                                    len(subtitle) > 0
                                    and max_line_count is not None
                                    and (long_pause or line_count >= max_line_count)
                                    or seg_break
                            ):
                                # subtitle break
                                yield subtitle
                                subtitle = []
                                line_count = 1
                            elif line_len > 0:
                                # line break
                                line_count += 1
                                timing["word"] = "\n" + timing["word"]
                            line_len = len(timing["word"].strip())
                        subtitle.append(timing)
                        last = timing["start"]
                    chunk_index += max_words_per_line
            if len(subtitle) > 0:
                yield subtitle

        if len(result["segments"]) > 0 and "words" in result["segments"][0]:
            for subtitle in iterate_subtitles():
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([word["word"] for word in subtitle])
                if highlight_words:
                    last = subtitle_start
                    all_words = [timing["word"] for timing in subtitle]
                    for i, this_word in enumerate(subtitle):
                        start = self.format_timestamp(this_word["start"])
                        end = self.format_timestamp(this_word["end"])
                        if last != start:
                            yield last, start, subtitle_text

                        yield start, end, "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                if j == i
                                else word
                                for j, word in enumerate(all_words)
                            ]
                        )
                        last = end
                else:
                    yield subtitle_start, subtitle_end, subtitle_text
        else:
            for segment in result["segments"]:
                segment_start = self.format_timestamp(segment["start"])
                segment_end = self.format_timestamp(segment["end"])
                segment_text = segment["text"].strip().replace("-->", "->")
                yield segment_start, segment_end, segment_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
                self.iterate_result(result, options, **kwargs), start=1
        ):
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
            self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        json.dump(result, file)


def get_writer(
        output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }
    os.makedirs(output_dir, exist_ok=True)
    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
                result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)


def generate_srt(words, punctuation, max_words, delay_threshold, output_file="output.srt"):
    """
    Generate and save an SRT file from words data.

    Parameters:
        words: List of dictionaries with 'start', 'end', 'word', and 'probability' keys.
        punctuation: List of punctuation characters to split on.
        max_words: Maximum number of words per segment.
        delay_threshold: Maximum delay between words to allow in a single segment.
        output_file: File name to save the SRT content.

    Returns:
        The name of the saved file.
    """

    def format_timestamp(
            seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
    ):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
        )

    segments = []
    current_segment = []
    segment_start = None

    for i, word in enumerate(words):
        if not current_segment:
            segment_start = word['start']

        if current_segment:
            # Check for punctuation split
            last_word = current_segment[-1]
            if (last_word['word'][-1] in punctuation and word['word'][0] == ' '):
                segments.append(current_segment)
                current_segment = []
                segment_start = word['start']

            # Check for delay split
            elif word['start'] - last_word['end'] > delay_threshold:
                segments.append(current_segment)
                current_segment = []
                segment_start = word['start']

        # Add the current word to the segment
        current_segment.append(word)

        # Check for max words split
        if len(current_segment) == max_words:
            segments.append(current_segment)
            current_segment = []
            segment_start = None if i == len(words) - 1 else words[i + 1]['start']

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    # Build the SRT content
    srt_content = ""
    for i, segment in enumerate(segments):
        start_time = format_timestamp(segment[0]['start'])
        end_time = format_timestamp(segment[-1]['end'])
        text = "".join([word['word'] for word in segment]).strip()  # Proper concatenation
        srt_content += f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n"

    # Save to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(srt_content)

    return output_file
