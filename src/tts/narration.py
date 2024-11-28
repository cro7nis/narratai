import os

import numpy as np
import srt
from scipy.io import wavfile

from tts.utils import check_if_extra_time_is_needed, find_available_space, expand_segment_if_possible, remove_tags, \
    add_silence, generate_full_audio_with_silence, read_srt, normalize
from utils.logger import logger
import uuid
from timeit import default_timer as timer
from tqdm import tqdm


class SubtitleNarrator:

    def __init__(self, generator, basic_speed=1.08, max_words_per_second=2.7, output_dir='outputs/'):
        super().__init__()
        self.generator = generator
        self.sr = self.generator.config.model_args.output_sample_rate
        self.basic_speed = basic_speed
        self.max_words_per_second = max_words_per_second

    def narrate(self, raw_subtitles=None, file=None, voice_speed=0.5, language='en', speaker_id=0, speaker_reference_file=None,
                speaker_data=None,
                output_dir='outputs/',  progress=None, initial_progress=0.0):
        if file:
            logger.info(f'Generating voice from: {file} file')
        os.makedirs(output_dir, exist_ok=True)
        total_iterations = 0
        audio_id = uuid.uuid4()
        if file is not None:
            raw_subtitles = read_srt(file)
        subtitles = list(srt.parse(raw_subtitles))
        audio_arrays = []
        start = timer()
        step = np.round((1.0-initial_progress) / len(subtitles), 2)

        if voice_speed is not None:
            voice_speed = normalize(voice_speed, 0.0, 1.0, -0.2, 0.25)
        else:
            voice_speed = 0.0
        logger.debug(f'voice_speed: {voice_speed}')

        for i, sub in tqdm(enumerate(subtitles)):
            logger.debug(f'{i}. ----------')
            text = remove_tags(sub.content)
            sub_duration = (sub.end - sub.start).total_seconds()
            logger.debug(subtitles[i])
            logger.debug(f'Duration: {sub_duration}')
            extra_seconds_needed, number_of_words = check_if_extra_time_is_needed(sub.content,
                                                                                  sub_duration,
                                                                                  words_per_second=
                                                                                  self.max_words_per_second)
            if extra_seconds_needed > 0:
                left_space, right_space = find_available_space(subtitles, i)
                expand_segment_if_possible(subtitles[i], left_space, right_space, extra_seconds_needed)
                sub_duration = (sub.end - sub.start).total_seconds()
                logger.debug(f'Potential expansion of the segment for {extra_seconds_needed} seconds.')
                logger.debug(f'Duration after expansion: {sub_duration}')

            # apply speed boost based on average word size
            average_word_length = np.clip(len(text) / number_of_words, 4, 9)
            speed_boost = normalize(average_word_length, 4, 9, 0.0, 0.1)
            logger.debug(f'average_word_length: {average_word_length} speed boost: {speed_boost}')



            # define initial speed based on max words in a specified interval
            max_words = self.max_words_per_second * sub_duration
            ratio = number_of_words / max_words
            clipped_ratio = np.clip(ratio, 0.5, 1.5)
            speed = normalize(clipped_ratio, 0.5, 1.5, 0.97, 1.28)
            speed = np.clip(speed + speed_boost, 0.97, 1.28)

            speed = speed + voice_speed

            logger.debug(f'Ratio: {ratio}')
            logger.debug(f'Clipped ratio: {clipped_ratio}')
            logger.debug(f'Initial Speed: {speed}')

            iterations = 0
            # speed = self.basic_speed
            start_sub = timer()
            while 1:
                if iterations > 30:
                    raise Exception('Cannot fit generated speech in SRT segment')
                if text == '':
                    x = np.zeros(int(sub_duration * self.sr))
                    logger.debug("NO TEXT")
                else:
                    x = self.generator.synthesize(text, speed, language=language,
                                                  speaker_reference_file=speaker_reference_file, speaker_id=speaker_id,
                                                  speaker_data=speaker_data)

                synthesized_audio_duration = len(x) / self.sr
                iterations += 1
                total_iterations += 1
                if synthesized_audio_duration - sub_duration > 0:
                    speed += 0.04
                else:
                    logger.debug(f'Performed {iterations} iterations with final speed: {speed}')
                    break
            logger.debug(
                f'Absolute difference of the duration '
                f'(without adding silence) is {np.abs(synthesized_audio_duration - sub_duration)} seconds')
            x = add_silence(x, self.sr, sub_duration)
            synthesized_audio_duration = len(x) / self.sr
            audio_arrays.append(x)
            logger.debug(f'Sub processing duration: {timer() - start_sub:.2f} seconds')
            if progress is not None:
                current_progress = np.clip(initial_progress + ((i+1) *step), initial_progress, 1.0)
                progress(current_progress, 'Generating Voice')

        final_audio_array = generate_full_audio_with_silence(subtitles, audio_arrays, sample_rate=self.sr)
        filepath = os.path.join(output_dir, f'{audio_id}.wav')
        wavfile.write(filepath, self.sr, final_audio_array)
        duration = timer() - start
        logger.info(
            f'Total iterations {total_iterations} for {len(subtitles)} subtitle'
            f' segments completed in {duration:.2f} seconds')
        logger.debug(f'Subs/iterations to ratio {len(subtitles) / total_iterations:.2f}')
        logger.info(f'Generated file: {filepath}')
        return filepath


def merge_adjacent_subtitles(srt_file_path, output_file_path, max_duration_seconds=10):
    # Load the subtitle file
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        subs = list(srt.parse(file.read()))

    # Initialize list for merged subtitles
    merged_subs = []
    new_index = 1  # To keep track of new subtitle index after merging

    # Variables for merging
    merged_text = ""
    merged_start = None
    merged_end = None

    for i, sub in enumerate(subs):
        # Determine if this subtitle is the start of a new merged subtitle
        if not merged_text:
            merged_start = sub.start
            merged_text = sub.content
            merged_end = sub.end
        else:
            # Append current subtitle text to merged text
            merged_text += " " + sub.content
            merged_end = sub.end  # Set end to the end of the current subtitle part

        # Calculate duration and check ending conditions
        segment_duration = (merged_end - merged_start).total_seconds()
        ends_with_sentence = merged_text.strip()[-1] in '.?!'
        ends_with_comma = merged_text.strip()[-1] == ','

        # Finalize the current merged segment if it ends with a sentence,
        # or is the last subtitle (but ignore cases where it ends with a comma) or exceeds max duration
        if (ends_with_sentence and not ends_with_comma) or segment_duration >= max_duration_seconds:
            # Create and store the merged subtitle with correct end time
            new_sub = srt.Subtitle(
                index=new_index,
                start=merged_start,
                end=merged_end,  # Use the actual end time of the last part in the merged segment
                content=merged_text.strip()
            )
            merged_subs.append(new_sub)
            new_index += 1

            # Reset for the next segment
            merged_text = ""
            merged_start = None
            merged_end = None

    # If there’s any leftover text after the loop, add it as a final subtitle
    if merged_text:
        new_sub = srt.Subtitle(
            index=new_index,
            start=merged_start,
            end=merged_end,
            content=merged_text.strip()
        )
        merged_subs.append(new_sub)

    # Save the modified subtitle file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(srt.compose(merged_subs))

    print(f"Merged subtitles saved to {output_file_path}")
