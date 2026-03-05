"""
translation_engine.py
=============================================================================
TranslationEngine Module - Translate transcribed text locally using NLLB-200.
=============================================================================
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    @abstractmethod
    def translate(self, text: Union[str, List[str]], src_lang: str, tgt_lang: str) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def free_memory(self) -> None:
        pass


class NLLBTranslator(BaseTranslator):
    """
    Implementation of the NLLB Translator using AutoModel classes for stability.
    Bypasses 'pipeline' task errors.
    """
    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        logger.info(f"Initializing NLLB Model: {self.model_id} on {self._device}...")
        
        try:
            # Suppress noisy transformers warnings
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, 
                dtype=self._dtype,
                low_cpu_mem_usage=True
            ).to(self._device)
            logger.info("NLLB Model and Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLLB model '{self.model_id}': {e}")
            if self._device == "cuda":
                logger.warning("Attempting CPU fallback...")
                self._device = "cpu"
                self._dtype = torch.float32
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to("cpu")
            else:
                raise

    def translate(self, text: Union[str, List[str]], src_lang: str = "eng_Latn", tgt_lang: str = "vie_Latn") -> Union[str, List[str]]:
        if not self.model or not self.tokenizer:
            return text
        
        if not text: return text
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            # Set source language
            self.tokenizer.src_lang = src_lang
            
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self._device)
            
            # Use inference_mode for maximum speed and lower memory
            with torch.inference_mode():
                # Generate translation with forced target language
                tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
                
                gen_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=512
                )
                
                outputs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            return outputs[0] if is_single else outputs
        except Exception as e:
            logger.error(f"NLLB Translation error: {e}")
            return text if is_single else [text]

    def free_memory(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TranslationEngine:
    _instance: Optional["TranslationEngine"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "TranslationEngine":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(TranslationEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._lock = threading.Lock()
        self._model_cache: Dict[str, BaseTranslator] = {}
        self.active_translator: Optional[BaseTranslator] = None
        self.active_model_type: Optional[str] = None

    def load_model(self, model_type: str = "nllb") -> None:
        with self._lock:
            if self.active_model_type == model_type and self.active_translator is not None:
                return

            if model_type in self._model_cache:
                self.active_translator = self._model_cache[model_type]
                self.active_model_type = model_type
                return

            if model_type == "nllb":
                translator = NLLBTranslator()
            else:
                raise ValueError(f"Unknown translation model type: {model_type}")

            self._model_cache[model_type] = translator
            self.active_translator = translator
            self.active_model_type = model_type

    def translate(self, text: Union[str, List[str]], src_lang: str, tgt_lang: str) -> Union[str, List[str]]:
        if self.active_translator is None:
            self.load_model("nllb")
        return self.active_translator.translate(text, src_lang, tgt_lang)

    def clear_cache(self) -> None:
        with self._lock:
            for translator in self._model_cache.values():
                translator.free_memory()
            self._model_cache.clear()
            self.active_translator = None
            self.active_model_type = None
            logger.info("Translation cache cleared.")
