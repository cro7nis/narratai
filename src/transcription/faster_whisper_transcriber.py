import os
from timeit import default_timer as timer
from typing import Optional

import faster_whisper
import torch
from faster_whisper.audio import decode_audio
from faster_whisper.utils import download_model, available_models

from transcription.transcriber import Transcriber
from transcription.utils.file import unpack
from transcription.utils.type import ASRResult
from utils.logger import logger


class FasterWhisperTranscriber(Transcriber):

    def __init__(self, cfg):
        super(FasterWhisperTranscriber, self).__init__(cfg)
        self.ckpt_downloaded = False
        self.save_dir = os.path.join(self.cfg.ckpt_path, self.cfg.model)
        self.download_whisper_checkpoint()
        self.name = 'FasterWhisperTranscriber'
        self.device = None

    def initialize_model(self) -> None:
        self.device = self.get_device(self.device)
        device_index = 0
        if ':' in self.device:
            device, device_index = self.device.split(':')
            device_index = int(device_index)
        else:
            device = self.device
        if self.device == 'cpu':
            if self.cfg.compute_type not in ['float32', 'int8']:
                self.logger.warning(f'{self.cfg.compute_type} not supported for cpu. Changing to float32')
                self.cfg.compute_type = 'float32'
        self.model = faster_whisper.WhisperModel(self.cfg.model, download_root=self.save_dir, device=device,
                                                 device_index=device_index,
                                                 compute_type=self.cfg.compute_type,
                                                 local_files_only=self.cfg.local_files_only,
                                                 num_workers=self.cfg.num_workers,
                                                 cpu_threads=self.cfg.cpu_threads)
        self.logger.info(f'Model is running on {self.device}')
        self.loaded = True

    def assert_checkpoint_is_downloaded(self):
        if not self.cfg.local_files_only:
            if not self.ckpt_downloaded:
                raise Exception('Checkpoint is still downloading. Please try again in a while')

    def download_whisper_checkpoint(self):
        if not self.cfg.local_files_only:
            if self.cfg.model in available_models():
                self.logger.info(f'Checking if ckpt exists exists in {self.save_dir}. If not it will download it')
                download_model(f'Systran/faster-whisper-{self.cfg.model}',
                               local_files_only=self.cfg.local_files_only,
                               cache_dir=self.save_dir)
                self.ckpt_downloaded = True

            else:
                raise RuntimeError(
                    f"Model {self.cfg.model} not found; available models = {available_models()}"
                )

    def transcribe(self, audio_file: str, language: str = None, task: str = 'transcribe'):
        assert task in ['transcribe', 'translate']
        result = self.transcribe_audio(audio_file, language=language, task=task)
        self.empty_cache()
        return result

    def transcribe_audio(self, audio_file: str, language: str = None, task: str = 'transcribe') -> ASRResult:
        logger.info(f'Transcribing {audio_file}')
        self.assert_checkpoint_is_downloaded()
        self.load_model_if_not_loaded()
        start = timer()
        audio = decode_audio(audio_file)
        segments, info = self.model.transcribe(audio, language=language, task=task, **self.cfg.parameters)
        segments = list(segments)
        segments = unpack(segments)
        asr_result = {'segments': segments, 'language': info.language}
        language = asr_result['language']

        if self.cfg.parameters.word_timestamps:
            words = []
            for i in asr_result['segments']:
                words.extend(i['words'])
            word_hyp = [i['word'] for i in words]
            word_ts_hyp = [[i['start'], i['end']] for i in words]
            transcription = ' '.join([segment['text'] for segment in asr_result['segments']]).strip()
        else:
            transcription = ' '.join([segment['text'] for segment in asr_result['segments']]).strip()
            word_hyp = None
            word_ts_hyp = None

        end = timer()
        processing_duration = end - start
        logger.info(f'Transcription completed in {processing_duration:.2f} seconds')

        return ASRResult(transcription=transcription, word_hyp=word_hyp, word_ts_hyp=word_ts_hyp,
                         processing_duration=processing_duration,
                         language=language, raw_result=asr_result)
