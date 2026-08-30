from piper.phonemize_ru_g2p import RuG2PPhonemizer


def test_ru_g2p_phonemizer() -> None:
    """Test the Russian RU_G2P-based phonemizer."""
    phonemizer = RuG2PPhonemizer()
    text = "Привет. Как дела?"
    phonemes = phonemizer.phonemize(text)
    assert phonemes == [
        ["P", "R0", "I", "V0", "E0", "T"],
        ["K", "A0", "G", "D0", "I", "L", "A"],
    ]
