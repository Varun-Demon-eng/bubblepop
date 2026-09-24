"""
=============================================================================
  AI TEXT & DOCUMENT DEEPFAKE DETECTION ENGINE
  Phase 4 - Stylometric & TF-IDF N-Gram Forensics
=============================================================================
  Module   : deepfake_text_detector.py
  Dataset  : optimization-hashira/ai-text-detection-dataset
  Modality : TEXT ONLY (completely isolated from audio, image, video pipelines)
=============================================================================
"""

import os
import sys
import math
import pickle
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    sys.exit("[FATAL] scikit-learn is not installed. Run: pip install scikit-learn")

logging.basicConfig(
    level=logging.INFO,
    format="[TEXT-FORENSICS] %(levelname)s | %(message)s",
)
log = logging.getLogger("text_forensics")


# ===========================================================================
# SECTION 1 - CONFIGURATION DATACLASS
# ===========================================================================

@dataclass
class TextForensicsConfig:
    max_features_word: int = 15_000
    max_features_char: int = 15_000
    ngram_range_word: Tuple[int, int] = (1, 3)
    ngram_range_char: Tuple[int, int] = (3, 5)
    weights_path: str = "text_model.pkl"
    supported_ext: tuple = (".txt", ".pdf", ".docx", ".doc", ".md", ".json", ".csv")


# ===========================================================================
# SECTION 2 - STYLOMETRIC FEATURE EXTRACTOR
# ===========================================================================

class StylometricExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts high-level human vs AI text stylometrics:
    - Perplexity / Entropy proxy
    - Burstiness (variance of sentence lengths)
    - Average word length
    - Punctuation & uppercase ratio
    - Vocabulary richness (Type-Token Ratio)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            text_str = str(text) if text else ""
            words = re.findall(r'\b\w+\b', text_str.lower())
            sentences = [s.strip() for s in re.split(r'[.!?]+', text_str) if s.strip()]

            n_words = len(words)
            n_chars = len(text_str)
            n_sents = len(sentences)

            # 1. Average Sentence Length
            avg_sent_len = n_words / max(n_sents, 1)

            # 2. Burstiness (Variance of sentence lengths)
            sent_lens = [len(s.split()) for s in sentences] if sentences else [0]
            burstiness = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0

            # 3. Average Word Length
            avg_word_len = sum(len(w) for w in words) / max(n_words, 1)

            # 4. Type-Token Ratio (Vocabulary Richness)
            ttr = len(set(words)) / max(n_words, 1)

            # 5. Punctuation & Capitalization Ratios
            punct_count = len(re.findall(r'[^\w\s]', text_str))
            punct_ratio = punct_count / max(n_chars, 1)
            upper_ratio = sum(1 for c in text_str if c.isupper()) / max(n_chars, 1)

            # 6. Entropy estimate (Character level entropy)
            char_counts = {}
            for c in text_str:
                char_counts[c] = char_counts.get(c, 0) + 1
            entropy = -sum((count / max(n_chars, 1)) * math.log2(count / max(n_chars, 1)) for count in char_counts.values())

            features.append([
                avg_sent_len,
                burstiness,
                avg_word_len,
                ttr,
                punct_ratio,
                upper_ratio,
                entropy,
            ])

        return np.array(features, dtype=np.float32)


# ===========================================================================
# SECTION 3 - TEXT INGESTION PIPELINE
# ===========================================================================

class TextIngestionPipeline:
    """Reads raw strings or document files (.txt, .md, .json, .pdf, .docx)."""

    def __init__(self, config: TextForensicsConfig) -> None:
        self.config = config

    def load(self, input_data: str) -> str:
        path = Path(input_data)
        if path.exists() and path.is_file():
            ext = path.suffix.lower()
            if ext in (".txt", ".md", ".json", ".csv"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(path))
                    return "\n".join([page.extract_text() or "" for page in reader.pages])
                except Exception:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
            elif ext in (".docx", ".doc"):
                try:
                    import docx
                    doc = docx.Document(str(path))
                    return "\n".join([p.text for p in doc.paragraphs])
                except Exception:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        return str(input_data)


# ===========================================================================
# SECTION 4 - FORENSIC REPORT DATACLASS
# ===========================================================================

@dataclass
class TextForensicReport:
    source: str
    verdict: str
    confidence: float
    prob_authentic: float
    prob_ai_generated: float
    character_count: int
    word_count: int


# ===========================================================================
# SECTION 5 - PRODUCTION INFERENCE ENGINE
# ===========================================================================

class TextDeepfakeDetector:
    """
    Public inference wrapper for AI Text Detection.
    Trained on 'optimization-hashira/ai-text-detection-dataset'.
    """

    def __init__(self, config: Optional[TextForensicsConfig] = None) -> None:
        self.config = config or TextForensicsConfig()
        self.ingestion = TextIngestionPipeline(self.config)
        self.model = self._load_or_build_model()

    def predict_text(self, text_or_path: str) -> TextForensicReport:
        raw_text = self.ingestion.load(text_or_path)
        if not raw_text.strip():
            return TextForensicReport(
                source=str(text_or_path)[:30],
                verdict="AUTHENTIC",
                confidence=0.90,
                prob_authentic=0.90,
                prob_ai_generated=0.10,
                character_count=0,
                word_count=0,
            )

        probs = self.model.predict_proba([raw_text])[0]
        prob_auth = float(probs[0])
        prob_ai   = float(probs[1])

        verdict = "AI-GENERATED" if prob_ai >= 0.50 else "AUTHENTIC"
        confidence = max(prob_auth, prob_ai)

        return TextForensicReport(
            source=str(text_or_path)[:40],
            verdict=verdict,
            confidence=confidence,
            prob_authentic=prob_auth,
            prob_ai_generated=prob_ai,
            character_count=len(raw_text),
            word_count=len(raw_text.split()),
        )

    def _load_or_build_model(self):
        weights_path = Path(self.config.weights_path)
        if weights_path.exists():
            try:
                with open(weights_path, "rb") as f:
                    model = pickle.load(f)
                log.info("Loaded trained AI Text Model from '%s'", weights_path.name)
                return model
            except Exception as exc:
                log.warning("Could not load '%s': %s. Building default pipeline.", weights_path.name, exc)

        # Default fallback pipeline
        return self.build_pipeline(self.config)

    @staticmethod
    def build_pipeline(config: TextForensicsConfig):
        word_vectorizer = TfidfVectorizer(
            ngram_range=config.ngram_range_word,
            max_features=config.max_features_word,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=config.ngram_range_char,
            max_features=config.max_features_char,
            sublinear_tf=True,
        )
        stylometric = StylometricExtractor()

        union = FeatureUnion([
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
            ("stylometrics", stylometric),
        ])

        classifier = LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced")

        pipeline = Pipeline([
            ("features", union),
            ("clf", classifier),
        ])
        return pipeline


# ===========================================================================
# SECTION 6 - TRAINING UTILITY FOR HUGGINGFACE DATASET
# ===========================================================================

def train_on_huggingface_dataset(save_path: str = "text_model.pkl", sample_limit: int = 50_000):
    """
    Trains TextDeepfakeDetector on 'optimization-hashira/ai-text-detection-dataset'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("[FATAL] datasets library not installed. Run: pip install datasets")

    log.info("Loading HuggingFace dataset 'optimization-hashira/ai-text-detection-dataset'...")
    dataset = load_dataset("optimization-hashira/ai-text-detection-dataset")

    train_data = dataset["train"]
    if sample_limit and len(train_data) > sample_limit:
        log.info("Subsampling dataset to %d rows for fast optimal training...", sample_limit)
        train_data = train_data.shuffle(seed=42).select(range(sample_limit))

    # Inspect dataset columns
    col_names = train_data.column_names
    log.info("Dataset Columns: %s", col_names)

    # Resolve text and label columns
    text_col = "text" if "text" in col_names else "content" if "content" in col_names else col_names[0]
    label_col = "label" if "label" in col_names else "generated" if "generated" in col_names else col_names[1]

    texts = train_data[text_col]
    labels = train_data[label_col]

    log.info("Extracted %d samples. Positive (AI) count: %d, Negative (Human) count: %d",
             len(texts), sum(1 for l in labels if l == 1), sum(1 for l in labels if l == 0))

    config = TextForensicsConfig(weights_path=save_path)
    pipeline = TextDeepfakeDetector.build_pipeline(config)

    log.info("Training Stylometric + N-Gram TF-IDF Classifier...")
    pipeline.fit(texts, labels)

    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)

    log.info("SUCCESS! AI Text Detection Model saved to '%s'", save_path)
    return pipeline


if __name__ == "__main__":
    train_on_huggingface_dataset("text_model.pkl")
