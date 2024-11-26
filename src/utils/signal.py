import logging
import os
from typing import Optional

import librosa
import numpy as np
from pydub import AudioSegment


def convert_2_wav(input_path, audio_path, sample_rate=16000, log=False):
    import subprocess
    # convert a file from mp3/mp4 to wav mono 16KHz
    flag = '-loglevel quiet' if not log else ''
    command = f'ffmpeg -i {input_path} -y -ar {sample_rate} -ac 1 {audio_path} ' + flag
    result = subprocess.call(command, shell=True)

def merge_video_and_audio(video_path, audio_path, output_video_name='merged', log=False):
    import subprocess
    dir_path, video_name = os.path.split(video_path)
    extension = video_name.split('.')[-1]
    flag = '-loglevel quiet' if not log else ''
    output_video_path = os.path.join(dir_path, output_video_name + '.' + extension)
    command  = f'ffmpeg -y -i {video_path} -i {audio_path} -c:v copy -map 0:v:0 -map 1:a:0 {output_video_path} ' + flag
    result = subprocess.call(command, shell=True)
    return output_video_path


def read_signal(path, sr=Optional[int], plot=False, log=False, mono=False):
    name = path.split('/')[-1]
    signal, sr = librosa.load(path, sr=sr, mono=mono)
    duration = librosa.get_duration(y=signal, sr=sr)
    if log:
        logging.info(f'Duration of audio is {duration} seconds')
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        fig.set_figwidth(20)
        fig.set_figheight(2)
        plt.plot(np.arange(len(signal)), signal, 'gray')
        fig.suptitle(name + ' audio', fontsize=16)
        plt.xlabel('time (secs)', fontsize=18)
        ax.margins(x=0)
        plt.ylabel('signal strength', fontsize=16);
        a, _ = plt.xticks();
        plt.xticks(a, a / sr);
    return signal, sr, duration

def clip_audio(input_audio_path, output_audio_path, start_time, end_time):
    """
    Clips a portion of the audio file based on the given start and end times.

    Parameters:
        input_audio_path: Path to the input audio file.
        output_audio_path: Path to save the clipped audio file.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        None
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_audio_path)

    # Convert start and end times to milliseconds
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    # Clip the audio
    clipped_audio = audio[start_ms:end_ms]

    # Export the clipped audio
    clipped_audio.export(output_audio_path, format="wav")  # You can change the format as needed
    # print(f"Clipped audio saved to {output_audio_path}")