import unicodedata
import re
import string
from typing import List

# CPU-light: no nltk download at import (was causing 1-2s delay per worker)
# Sentence split via regex fallback, deterministic and no model load

class Tokenizer:
    """
    PRD §2 + §5: Normalization -> sentence boundary -> tokenization
    - Unicode NFKC (PRD §2)
    - Sentence segmentation for hierarchical chunking (PRD §12)
    - Generates 1-token, 2-token phrase, 3-8 token context windows (PRD §5)
    """
    def __init__(self, lowercase: bool = True, keep_stopwords: bool = True):
        # PRD §9-10: cross-lingual - jangan agresif drop stopwords karena
        # "bagaimana perusahaan meningkatkan pendapatan" -> kita butuh semua token untuk semantic lookup
        # keep_stopwords=True -> preserve for semantic path, filter only for lexical BM25
        self.lowercase = lowercase
        self.keep_stopwords = keep_stopwords

    def normalize(self, text: str) -> str:
        # PRD §2: Unicode normalization
        text = unicodedata.normalize("NFKC", text)
        # normalize whitespace, keep sentence delimiters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def sentences(self, text: str) -> List[str]:
        text = self.normalize(text)
        # CPU-light: regex split, no nltk (avoid punkt model load on CPU)
        # Handles ID/EN + preserves CJK (no sentence delimiter)
        return re.split(r"(?<=[.!?])\s+", text)

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        if self.lowercase:
            text = text.lower()
        # PRD: tokenization - keep alphanumeric, split on punctuation but preserve for phrase codes
        # Multilingual friendly regex; TIDAK memecah angka berformat ribuan "62.714.280" / "1.577.780"
        # (penting untuk dokumen finansial — angka adalah token kunci, generik bukan hard-coded)
        tokens = re.findall(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\w+", text, flags=re.UNICODE)
        # filter single-char noise but keep CJK etc?
        tokens = [t for t in tokens if len(t) > 1 or t.isalnum()]
        return tokens

    def lexical_tokens(self, text: str) -> List[str]:
        """For BM25 lexical index - lowercase + punctuation removed (existing behavior)"""
        return self.tokenize(text)

    def contextual_windows(self, tokens: List[str], n_min: int = 1, n_max: int = 8):
        """
        PRD §5: Contextual semantic coding
        Generate 1-token code, 2-token phrase code, 3-8 token context code via sliding window
        Returns list of windows: [(ngram_tuple, position)]
        """
        windows = []
        for n in range(n_min, min(n_max + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                windows.append((tuple(tokens[i:i+n]), i))
        return windows
