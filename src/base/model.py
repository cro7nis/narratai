from abc import abstractmethod, ABC
from timeit import default_timer as timer

import torch
from utils.logger import logger


class Model(ABC):

    def __init__(self, device=None):
        self.logger = logger.bind(classname=self.__class__.__name__)
        self.loaded = False
        self.model = None
        self.name = self.__class__.__name__
        self.device = device

    @abstractmethod
    def initialize_model(self) -> None:
        raise NotImplementedError

    def empty_cache(self):
        if 'cuda' in self.device:
            torch.cuda.empty_cache()

    @staticmethod
    def get_device(device):
        if device is not None:
            if 'cuda' in device:
                if torch.cuda.is_available():
                    return device
                else:
                    return 'cpu'
            elif 'cpu' in device:
                return 'cpu'
        else:
            if torch.cuda.is_available():
                return 'cuda:0'
            else:
                return 'cpu'

    def load_model_if_not_loaded(self):
        if not self.loaded:
            self.logger.info(f'Loading {self.name} model')
            start = timer()
            self.initialize_model()
            self.logger.info(f'{self.name} model loaded to {self.device} after {timer() - start:.2f} seconds')
            self.loaded = True
