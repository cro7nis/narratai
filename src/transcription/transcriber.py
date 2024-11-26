from typing import Optional

import torch
from omegaconf import OmegaConf
from timeit import default_timer as timer
from abc import ABC, abstractmethod

from base.model import Model
from utils.logger import logger
from transcription.utils.type import ASRResult, LanguageResult
import numpy as np


class Transcriber(Model):

    def __init__(self, cfg):
        if isinstance(cfg, str):
            self.cfg = OmegaConf.load(cfg)
        else:
            self.cfg = cfg
        super().__init__(self.cfg.device if 'device' in self.cfg else None)

    @abstractmethod
    def transcribe(self, audio_file: str, language: str = None, speaker_labels: list = None, task: str = 'transcribe',
                   available_languages: list[str] = None, language_mode: Optional[str] = 'suggest') -> ASRResult:
        pass

    @abstractmethod
    def detect_language(self, audio_file, fallback_language=None) -> LanguageResult:
        pass

    def detect_languages(self, audios, original_audio, sampling_rate, confidence_threshold=0.8, fallback_language=None,
                         short_duration_threshold=8, padding=5):
        confident_languages = set()
        language_results = []
        total_duration = original_audio.shape[0] / sampling_rate
        for audio in audios:
            audio_segment = audio['array']
            if audio['duration'] < short_duration_threshold:
                audio_segment = original_audio[
                                int(np.clip(float(audio['start'] - padding), 0, total_duration) * sampling_rate)
                                :int(np.clip(float(audio['stop'] + padding), 0, total_duration) * sampling_rate)]
            lang_res = self.detect_language(audio_segment, fallback_language=fallback_language)
            language_results.append(lang_res)
        self.logger.debug(f'Languages detected {language_results}')
        language_list = []
        max_probability = 0.0
        set_fallback_language = fallback_language is None

        for lang_res in language_results:
            if lang_res.probability > confidence_threshold:
                confident_languages.add(lang_res.language)
            if set_fallback_language:
                if lang_res.probability > max_probability:
                    fallback_language = lang_res.language
                    max_probability = lang_res.probability
        self.logger.debug(f'Confident languages {confident_languages}')
        self.logger.debug(f'Fallback language {fallback_language}')

        for i, lang_res in enumerate(language_results):
            added = False
            for pred_lang, prob in lang_res.predictions.items():
                if pred_lang in confident_languages and not added:
                    language_list.append(pred_lang)
                    added = True
            for pred_lang, prob in lang_res.predictions.items():
                if pred_lang == fallback_language and not added:
                    language_list.append(pred_lang)
                    added = True
            if not added:
                language_list.append(lang_res.language)

        self.logger.debug(f'Languages detected after fallback {language_list}')
        return language_list
