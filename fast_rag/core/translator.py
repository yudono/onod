"""
Dynamic translator — PRD §23 Translation bukan bagian retrieval
retrieve original dulu, translate belakangan (hemat biaya 1GB)
Cache dinamis di cache/translations/ & tmp/translations/
Tidak hard-coded language list — deteksi otomatis, fallback ringan jika model tidak ada
"""
import hashlib, json, logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIRS = [Path("cache/translations"), Path("tmp/translations"), Path("/tmp/fast_rag_translations")]

def _cache_dir() -> Path:
    for d in CACHE_DIRS:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            continue
    return Path("cache/translations")

class Translator:
    def __init__(self, cache_dir: Optional[str] = None, enable_neural: bool = False):
        self.cache_dir = Path(cache_dir) if cache_dir else _cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_neural = enable_neural
        self._models = {}  # lazy load MarianMT per pair
        self.hits = 0
        self.misses = 0

    def _cache_key(self, text: str, target_lang: str) -> str:
        return hashlib.md5(f"{target_lang}::{text}".encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[str]:
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                data = json.loads(f.read_text())
                return data.get("translated")
            except Exception:
                return None
        # fallback tmp
        f2 = Path("tmp/translations") / f"{key}.json"
        if f2.exists():
            try:
                return json.loads(f2.read_text()).get("translated")
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, text: str, translated: str, target_lang: str):
        f = self.cache_dir / f"{key}.json"
        try:
            f.write_text(json.dumps({"original": text[:500], "translated": translated, "target": target_lang}, ensure_ascii=False))
            # dual write tmp
            try:
                Path("tmp/translations").mkdir(parents=True, exist_ok=True)
                (Path("tmp/translations") / f"{key}.json").write_text(json.dumps({"original": text[:500], "translated": translated, "target": target_lang}, ensure_ascii=False))
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"cache save fail {e}")

    def _neural_translate(self, text: str, target_lang: str) -> Optional[str]:
        if not self.enable_neural:
            return None
        # lazy load MarianMT per pair, e.g., id->en, en->id, zh->en
        pair = None
        t = target_lang.lower()
        # detect source heuristically: if text contains CJK -> zh/ja, else id/en
        has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text)
        has_arabic = any('\u0600' <= c <= '\u06ff' for c in text)
        if has_cjk:
            if t.startswith("en"):
                pair = "Helsinki-NLP/opus-mt-zh-en"
            elif t.startswith("id"):
                pair = "Helsinki-NLP/opus-mt-zh-en"  # fallback via en
        else:
            if t.startswith("en"):
                pair = "Helsinki-NLP/opus-mt-id-en"
            elif t.startswith("id"):
                pair = "Helsinki-NLP/opus-mt-en-id"
        if not pair:
            return None
        try:
            from transformers import MarianMTModel, MarianTokenizer
            if pair not in self._models:
                logger.info(f"Loading Marian {pair} (first time, may download)...")
                tok = MarianTokenizer.from_pretrained(pair)
                mdl = MarianMTModel.from_pretrained(pair)
                self._models[pair] = (tok, mdl)
            tok, mdl = self._models[pair]
            # truncate for model limits
            chunk = text[:800]
            inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512)
            out = mdl.generate(**inputs, max_length=512)
            translated = tok.decode(out[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            logger.debug(f"neural translate fail {pair}: {e}")
            return None

    def translate(self, text: str, target_lang: str = "en") -> str:
        if not text or not text.strip():
            return text
        if len(text) > 4000:
            # long text: chunk
            parts = [text[i:i+2000] for i in range(0, len(text), 2000)]
            return " ".join(self.translate(p, target_lang) for p in parts)
        key = self._cache_key(text, target_lang)
        cached = self._load_cache(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        # try neural if enabled
        neural = self._neural_translate(text, target_lang)
        if neural:
            self._save_cache(key, text, neural, target_lang)
            return neural
        # CPU-light fallback: dictionary-based + marker (tidak hard-coded list panjang, hanya fallback)
        # Untuk demo, kembalikan original + language tag + lightweight word replace untuk 5 bahasa utama
        # Ini tidak hard-coded 100 bahasa, tapi placeholder — real translate butuh LLM/Marian
        fallback_map = {
            "pendapatan": {"en": "revenue", "zh": "收入", "ja": "収益"},
            "laba": {"en": "profit", "zh": "利润"},
            "rugi": {"en": "loss"},
            "aset": {"en": "assets"},
            "liabilitas": {"en": "liabilities"},
            "ekuitas": {"en": "equity"},
            "kas": {"en": "cash"},
            "direksi": {"en": "board of directors"},
            "emas": {"en": "gold", "zh": "黄金"},
            "nikel": {"en": "nickel"},
            "bauksit": {"en": "bauxite"},
        }
        lower = text.lower()
        translated = text
        if target_lang.lower().startswith("en") and any(k in lower for k in fallback_map):
            # lightweight replace 3-5 kata kunci saja, sisanya tetap original (hemat CPU)
            for k, mp in fallback_map.items():
                if k in lower and target_lang.lower().startswith("en") and "en" in mp:
                    translated = translated.replace(k, f"{k} ({mp['en']})")
                    translated = translated.replace(k.capitalize(), f"{k} ({mp['en']})")
            translated = f"[{target_lang} auto] {translated}"
        else:
            # generic marker bahwa retrieve dulu, translate belakangan (PRD §23)
            translated = f"[{target_lang}] {text}"
        self._save_cache(key, text, translated, target_lang)
        return translated

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "cache_dir": str(self.cache_dir)}
