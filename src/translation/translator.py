from abc import abstractmethod
from functools import lru_cache
from timeit import default_timer as timer
from typing import List, Union
from typing import Optional, Callable

from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

from base.model import Model
from translation.utils import iso2flores_code
from utils.logger import logger


class Translator(Model):

    def __init__(self, model_id: str,
                 text_preprocessor: Callable,
                 max_input_tokens_per_sentence: int = 210,
                 max_gen_tokens: int = 210,
                 unk_token: str = '<unk>',
                 special_token: str = '▁',
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 local_files_only: bool = True
                 ) -> None:
        super().__init__(device)
        self.model_id = model_id
        self.text_preprocessor = text_preprocessor
        self.max_input_tokens_per_sentence = max_input_tokens_per_sentence
        self.max_gen_tokens = max_gen_tokens
        self.unk_token = unk_token
        self.special_token = special_token
        self.cache_dir = cache_dir
        self.config = None
        self.tokenizer = None
        self.logger = logger.bind(classname=self.__class__.__name__)
        self.name = self.__class__.__name__
        self.language_list = None
        self.local_files_only = local_files_only

    @abstractmethod
    def initialize_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_tokenizer(self, src_lang):
        raise NotImplementedError

    @abstractmethod
    def check_languages(self, src_lang, tgt_lang):
        raise NotImplementedError

    def translate_batch(self, sentences: List[str], tokenizer, tgt_lang):
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=self.max_gen_tokens)
        output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return output

    def translate(self, text: Union[str, list[str]], src_lang, tgt_lang,
                  batch_size: int, return_sentences=False) -> Union[str, List[str]]:
        self.load_model_if_not_loaded()
        src_lang, tgt_lang = self.check_languages(src_lang, tgt_lang)
        start = timer()
        tokenizer = self.get_tokenizer(src_lang=src_lang)
        end = timer()
        duration = round(end - start, ndigits=2)
        self.logger.debug(f'get_tokenizer duration {duration}')
        start = timer()
        if isinstance(text, str):
            sentences_or_phrases = self.text_preprocessor(text, tokenizer, special_token=self.special_token,
                                                          max_tokens=self.max_input_tokens_per_sentence,
                                                          unk_token=self.unk_token)
        else:
            sentences_or_phrases = text
        end = timer()
        duration = round(end - start, ndigits=2)
        self.logger.debug(f'Text preprocessing duration {duration}')
        result = []
        batch = []
        start = timer()
        for sentence in sentences_or_phrases:
            batch.append(sentence)
            if len(batch) >= batch_size:
                result.extend(self.translate_batch(batch, tokenizer, tgt_lang))
                batch.clear()
        if batch:
            result.extend(self.translate_batch(batch, tokenizer, tgt_lang))
        end = timer()
        duration = round(end - start, ndigits=2)
        self.empty_cache()
        logger.info(f'Translation completed in {duration:.2f} seconds')

        if return_sentences:
            return result
        else:
            return ' '.join(result)


class NLLBTranslator(Translator):

    def __init__(self, model_id: str,
                 text_preprocessor: Callable,
                 max_input_tokens_per_sentence: int = 210,
                 max_gen_tokens: int = 210,
                 unk_token: str = '<unk>',
                 special_token: str = '▁',
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 local_files_only: bool = True
                 ) -> None:

        super().__init__(model_id, text_preprocessor, max_input_tokens_per_sentence, max_gen_tokens, unk_token,
                         special_token, cache_dir, device, local_files_only)
        self.logger = logger.bind(classname=self.__class__.__name__)
        self.name = self.__class__.__name__

    def check_languages(self, src_lang, tgt_lang):
        if src_lang not in iso2flores_code or tgt_lang not in iso2flores_code:
            self.logger.error(
                f'source_language {src_lang} and target_language {tgt_lang}  should'
                f' be in one of the following: {list(iso2flores_code.keys())}')
            raise Exception(f'source_language {src_lang} and target_language {tgt_lang}  should'
                            f' be in one of the following: {list(iso2flores_code.keys())}')
        else:
            return iso2flores_code[src_lang], iso2flores_code[tgt_lang]

    def initialize_model(self) -> None:
        self.device = self.get_device(self.device)
        self.config = AutoConfig.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                 local_files_only=self.local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                           local_files_only=self.local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.language_list = list(iso2flores_code.keys())

    @lru_cache(maxsize=256)
    def get_tokenizer(self, src_lang):
        start = timer()
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=src_lang, cache_dir=self.cache_dir,
                                                  local_files_only=self.local_files_only)
        end = timer()
        duration = round(end - start, ndigits=2)
        self.logger.info(
            f"Loaded tokenizer for {self.model_id}; src_lang={src_lang} after {duration} seconds")
        return tokenizer


class M2M100Translator(Translator):

    def __init__(self, model_id: str,
                 text_preprocessor: Callable,
                 max_input_tokens_per_sentence: int = 210,
                 max_gen_tokens: int = 210,
                 unk_token: str = '<unk>',
                 special_token: str = '▁',
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 local_files_only: bool = True
                 ) -> None:

        super().__init__(model_id, text_preprocessor, max_input_tokens_per_sentence, max_gen_tokens, unk_token,
                         special_token, cache_dir, device, local_files_only)
        self.logger = logger.bind(classname=self.__class__.__name__)
        self.name = self.__class__.__name__

    def check_languages(self, src_lang, tgt_lang):
        if src_lang not in self.language_list or tgt_lang not in self.language_list:
            self.logger.error(
                f'source_language {src_lang} and target_language {tgt_lang}  should'
                f' be in one of the following: {self.language_list}')
            raise Exception(f'source_language {src_lang} and target_language {tgt_lang}  should'
                            f' be in one of the following: {list(iso2flores_code.keys())}')
        else:
            return src_lang, tgt_lang

    def initialize_model(self) -> None:
        self.device = self.get_device(self.device)
        self.config = AutoConfig.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                 local_files_only=self.local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                       local_files_only=self.local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                           local_files_only=self.local_files_only)
        self.model.to(self.device)
        self.model.eval()
        self.language_list = list(self.tokenizer.lang_code_to_id.keys())

    def get_tokenizer(self, src_lang):
        self.tokenizer.src_lang = src_lang
        return self.tokenizer
