"""BGE-M3 임베딩 모델을 로딩하고 텍스트를 벡터로 변환하는 모듈."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-m3"


@dataclass(frozen=True)
class EmbeddingResult:
    """텍스트 1개를 임베딩한 결과를 설명하는 데이터 구조."""

    text: str
    vector: np.ndarray
    dimension: int
    norm: float


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Hugging Face 모델 이름 또는 로컬 경로로 Sentence Transformers 모델을 로딩한다."""

    return SentenceTransformer(model_name)


def embed_text(
    model: SentenceTransformer,
    text: str,
    *,
    normalize_embeddings: bool = False,
) -> EmbeddingResult:
    """텍스트 하나를 임베딩 벡터로 변환하고 입력·벡터·차원·norm을 반환한다.

    Args:
        model: ``load_embedding_model``이 반환한 임베딩 모델.
        text: 임베딩할 문자열. 빈 문자열은 허용하지 않는다.
        normalize_embeddings: True이면 L2 norm이 1이 되도록 정규화한다.
    """

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("임베딩할 텍스트를 입력해 주세요.")

    encoded = model.encode(
        cleaned_text,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    vector = np.asarray(encoded, dtype=np.float32).reshape(-1)

    return EmbeddingResult(
        text=cleaned_text,
        vector=vector,
        dimension=int(vector.shape[0]),
        norm=float(np.linalg.norm(vector)),
    )
