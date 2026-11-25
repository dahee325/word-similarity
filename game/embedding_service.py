"""게임용 텍스트 임베딩 모듈. UI 코드와 모델 코드를 분리한다."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """모델 이름 또는 로컬 경로로 Sentence Transformers 모델을 로딩한다."""

    return SentenceTransformer(model_name)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """텍스트 하나를 1차원 NumPy 임베딩 벡터로 변환한다."""

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("관련 단어를 입력해 주세요.")
    encoded = model.encode(
        cleaned_text,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return np.asarray(encoded, dtype=np.float32).reshape(-1)
