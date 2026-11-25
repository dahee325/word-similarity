"""BGE-M3, Word2Vec, FastText의 임베딩과 유사도를 비교하는 모듈."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from gensim.models import FastText, Word2Vec

from embedding_service import EmbeddingResult, embed_text
from similarity_service import SimilarityResult, calculate_cosine_similarity


TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+")


@dataclass(frozen=True)
class TraditionalEmbeddingResult:
    """Word2Vec/FastText 임베딩 결과와 실제 사용 토큰을 보관한다."""

    text: str
    vector: np.ndarray
    dimension: int
    norm: float
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class ModelComparisonResult:
    """모델 하나의 두 텍스트 비교 결과."""

    model_name: str
    similarity: SimilarityResult
    first_embedding: TraditionalEmbeddingResult | EmbeddingResult
    second_embedding: TraditionalEmbeddingResult | EmbeddingResult


def tokenize(text: str) -> list[str]:
    """공백과 문장부호를 기준으로 실습용 토큰 목록을 만든다."""

    return TOKEN_PATTERN.findall(text.lower())


def prepare_sentences(sentences: Sequence[str]) -> list[list[str]]:
    """문장 모음을 gensim 학습용 토큰 리스트의 리스트로 변환한다."""

    prepared = [tokenize(sentence) for sentence in sentences]
    return [tokens for tokens in prepared if tokens]


def train_word2vec(
    sentences: Sequence[str],
    *,
    vector_size: int = 100,
    window: int = 3,
    epochs: int = 100,
    seed: int = 42,
) -> Word2Vec:
    """주어진 말뭉치로 Word2Vec 모델을 학습한다."""

    return Word2Vec(
        sentences=prepare_sentences(sentences),
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        seed=seed,
        epochs=epochs,
    )


def train_fasttext(
    sentences: Sequence[str],
    *,
    vector_size: int = 100,
    window: int = 3,
    epochs: int = 100,
    seed: int = 42,
) -> FastText:
    """주어진 말뭉치로 문자 n-gram을 활용하는 FastText 모델을 학습한다."""

    return FastText(
        sentences=prepare_sentences(sentences),
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=1,
        seed=seed,
        epochs=epochs,
    )


def embed_traditional_text(model: Any, text: str) -> TraditionalEmbeddingResult:
    """gensim 모델로 단어 또는 짧은 문장을 평균 임베딩한다."""

    tokens = tokenize(text)
    if not tokens:
        raise ValueError("비교할 단어를 입력해 주세요.")

    vectors: list[np.ndarray] = []
    missing: list[str] = []
    for token in tokens:
        try:
            vectors.append(np.asarray(model.wv.get_vector(token), dtype=np.float32))
        except KeyError:
            missing.append(token)

    if missing:
        raise ValueError(f"Word2Vec 말뭉치에 없는 토큰입니다: {', '.join(missing)}")

    vector = np.mean(vectors, axis=0).astype(np.float32)
    return TraditionalEmbeddingResult(
        text=text.strip(),
        vector=vector,
        dimension=int(vector.shape[0]),
        norm=float(np.linalg.norm(vector)),
        tokens=tuple(tokens),
    )


def compare_vectors(
    model_name: str,
    first: EmbeddingResult | TraditionalEmbeddingResult,
    second: EmbeddingResult | TraditionalEmbeddingResult,
) -> ModelComparisonResult:
    """이미 계산한 두 임베딩 결과를 코사인 유사도로 비교한다."""

    if isinstance(first, EmbeddingResult) and isinstance(second, EmbeddingResult):
        similarity = calculate_cosine_similarity(first, second)
    else:
        first_as_embedding = EmbeddingResult(first.text, first.vector, first.dimension, first.norm)
        second_as_embedding = EmbeddingResult(second.text, second.vector, second.dimension, second.norm)
        similarity = calculate_cosine_similarity(first_as_embedding, second_as_embedding)
    return ModelComparisonResult(model_name, similarity, first, second)


def compare_bge(model: Any, first_text: str, second_text: str) -> ModelComparisonResult:
    """BGE-M3로 두 텍스트를 임베딩하고 비교한다."""

    first = embed_text(model, first_text)
    second = embed_text(model, second_text)
    return compare_vectors("BGE-M3", first, second)
