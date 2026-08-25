"""Russian phonemization using RU_G2P (russian_g2p)."""

import inspect
import sys

# --- 1. Python 3.11+ / inspect & Python 3.12+ dawg / dawg2 Compatibility Patch ---
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple

    ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)

    inspect.getargspec = getargspec

# If `dawg` module is missing (e.g. Python 3.12+ where `dawg` fails to build due to `longintrepr.h`),
# alias `dawg2` (or `DAWG`) to `dawg` if available.
try:
    import dawg  # type: ignore # noqa: F401
except ImportError:
    try:
        import dawg2
        sys.modules["dawg"] = dawg2
    except ImportError:
        try:
            import DAWG
            sys.modules["dawg"] = DAWG
        except ImportError:
            pass

# --- 2. Keras 3 / TensorFlow 2.16+ Compatibility Patch ---
try:
    import json
    import keras
    import yaml

    custom_objects = keras.saving.get_custom_objects()
    if "Model" not in custom_objects:
        custom_objects["Model"] = keras.src.models.Functional
    if "Functional" not in custom_objects:
        custom_objects["Functional"] = keras.src.models.Functional
    for name in [
        "LSTM",
        "Dense",
        "Dropout",
        "Embedding",
        "InputLayer",
        "Reshape",
        "Concatenate",
        "TimeDistributed",
        "Bidirectional",
        "BatchNormalization",
        "Activation",
    ]:
        if name not in custom_objects:
            custom_objects[name] = getattr(keras.layers, name)

    def model_from_yaml(yaml_string, custom_objects_dict=None):
        config = yaml.unsafe_load(yaml_string)
        json_string = json.dumps(config)
        if custom_objects_dict is None:
            custom_objects_dict = {}
        custom_objects_dict["Model"] = keras.src.models.Functional
        custom_objects_dict["Functional"] = keras.src.models.Functional
        for k, v in keras.saving.get_custom_objects().items():
            if k not in custom_objects_dict:
                custom_objects_dict[k] = v
        return keras.models.model_from_json(json_string, custom_objects=custom_objects_dict)

    if not hasattr(keras.models, "model_from_yaml"):
        keras.models.model_from_yaml = model_from_yaml
        sys.modules["keras.models"] = keras.models
except Exception:
    pass


import re
from typing import List

from russian_g2p.Accentor import Accentor
from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme


class RuG2PPhonemizer:
    """Phonemizer that uses nsu-ai/russian_g2p."""

    def __init__(self) -> None:
        """Initialize Accentor and Grapheme2Phoneme models."""
        self.accentor = Accentor()
        self.g2p = Grapheme2Phoneme()

    def phonemize(self, text: str) -> List[List[str]]:
        """Text to Cyrillic-based phoneme lists grouped by sentence."""
        sentences = re.split(r"(?<=[.?!])\s+", text.strip())
        all_phonemes: List[List[str]] = []

        for sentence in sentences:
            if not sentence:
                continue

            sentence = re.sub(r"\s+", " ", sentence)
            dst = re.sub(r"[\.\,\?\!\(\);:]+", " <sil>", sentence.lower())
            dst = re.sub(r" [–-] |\n", " <sil> ", dst)
            dst = re.sub(r"\s{2,}", " ", dst)
            dst = re.sub(r"^\s|(?<!\w)[\\\/@#~¬`£€\$%\^\&\*–_=+\'\"\|«»–-]+", "", dst)

            tokens = dst.strip().split(" ")
            words_and_tags = [[tok] for tok in tokens if tok]
            if not words_and_tags:
                continue

            try:
                accented_text = self.accentor.do_accents(words_and_tags)
            except Exception:
                accented_text = []

            if accented_text:
                tmp = " " + " ".join(accented_text[0])
                phonetic_words = tmp.split(" <sil>")
                sentence_phonemes: List[str] = []
                for phonetic_word in phonetic_words:
                    if phonetic_word.strip():
                        try:
                            phonemes = self.g2p.phrase_to_phonemes(phonetic_word)
                            sentence_phonemes.extend(phonemes)
                        except Exception:
                            pass
                if sentence_phonemes:
                    all_phonemes.append(sentence_phonemes)

        return all_phonemes
