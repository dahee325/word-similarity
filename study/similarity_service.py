"""임베딩 벡터 두 개의 코사인 유사도를 계산하는 학습용 모듈."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedding_service import EmbeddingResult


@dataclass(frozen=True)
class SimilarityResult:
    """두 텍스트와 벡터 연산 결과를 함께 보관하는 데이터 구조."""

    first_text: str
    second_text: str
    cosine_similarity: float
    score_0_to_100: float
    first_dimension: int
    second_dimension: int
    first_norm: float
    second_norm: float
    dot_product: float


def calculate_cosine_similarity(
    first: EmbeddingResult,
    second: EmbeddingResult,
) -> SimilarityResult:
    """두 임베딩 결과의 코사인 유사도와 중간 벡터 연산 결과를 반환한다.

    코사인 유사도 공식:

        cosine(A, B) = (A · B) / (||A|| × ||B||)

    ``score_0_to_100``은 학습 화면에서 보기 쉽도록 원시 값(-1~1)을
    0~100 범위로 선형 변환한 값이다. 실제 모델 결과는 ``cosine_similarity``다.
    """

    if first.dimension != second.dimension:
        raise ValueError("두 임베딩 벡터의 차원이 다릅니다.")
    if first.norm == 0 or second.norm == 0:
        raise ValueError("L2 norm이 0인 벡터는 비교할 수 없습니다.")

    dot_product = float(np.dot(first.vector, second.vector))
    cosine = float(dot_product / (first.norm * second.norm))
    score = float(np.clip((cosine + 1.0) * 50.0, 0.0, 100.0))

    return SimilarityResult(
        first_text=first.text,
        second_text=second.text,
        cosine_similarity=cosine,
        score_0_to_100=score,
        first_dimension=first.dimension,
        second_dimension=second.dimension,
        first_norm=first.norm,
        second_norm=second.norm,
        dot_product=dot_product,
    )
