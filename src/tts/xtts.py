from TTS.utils.manage import ModelManager
import os
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import pickle

from base.model import Model
from utils.logger import logger
from tts.utils import remove_tags


class XttsGenerator(Model):
    SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'hu', 'ko',
                           'ja', 'hi']

    def __init__(self, model_name, device, checkpoint_path='checkpoints/',
                 speaker_dict_path='checkpoints/data/speaker_data.pkl'):
        super().__init__(device)
        self.model_name = model_name
        self.config = XttsConfig()
        self.checkpoint_path = checkpoint_path
        self.speaker_dict_path = speaker_dict_path
        with open(self.speaker_dict_path, 'rb') as f:
            self.speaker_data = pickle.load(f)
        self.speakers = list(self.speaker_data.keys())[-20:]

    def initialize_model(self) -> None:
        self.device = self.get_device(self.device)
        model_path, _, _ = ModelManager(output_prefix=self.checkpoint_path).download_model(self.model_name)
        logger.info(f'Model downloaded in {model_path}')
        self.config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_path=os.path.join(model_path, "model.pth"),
            vocab_path=os.path.join(model_path, "vocab.json"),
            eval=True,
            use_deepspeed=False,
        )
        self.model.to(self.device)

    def synthesize(self, text, speed=1.1, language='en', speaker_reference_file=None, speaker_id=0, speaker_data=None):
        self.load_model_if_not_loaded()
        if language == 'zh':
            language = 'zh-cn'
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(f'Not Supported Language {language}')
        if speaker_reference_file is not None:
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_reference_file)
        elif speaker_data is not None:
            gpt_cond_latent, speaker_embedding = speaker_data
        else:
            if isinstance(speaker_id, int):
                speaker = self.speakers[speaker_id]
            elif isinstance(speaker_id, str):
                speaker = speaker_id
            else:
                raise ValueError('speaker_id must be int or str')
            gpt_cond_latent = self.speaker_data[speaker]['gpt_cond_latent']
            speaker_embedding = self.speaker_data[speaker]['speaker_embedding']

        output = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=self.config.temperature,  # Add custom parameters here
            length_penalty=self.config.length_penalty,
            repetition_penalty=self.config.repetition_penalty,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            speed=speed
        )
        self.empty_cache()
        return output['wav']
