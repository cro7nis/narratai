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
    def transcribe(self, audio_file: str, language: str = None, task: str = 'transcribe') -> ASRResult:
        pass
