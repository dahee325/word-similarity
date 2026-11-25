"""임베딩 벡터 사이의 코사인 유사도 계산 모듈."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityResult:
    """코사인 유사도와 게임 표시 점수를 함께 보관한다."""

    cosine_similarity: float
    score: float


def calculate_cosine_similarity(
    target_vector: np.ndarray,
    clue_vector: np.ndarray,
) -> SimilarityResult:
    """두 벡터의 실제 코사인 유사도와 0~100 표시 점수를 계산한다.

    원시 코사인은 -1~1 값이며, 표시 점수는 ``(cosine + 1) * 50``이다.
    이는 모델 결과를 0~100 범위로 선형 표시하는 변환이다.
    """

    target = np.asarray(target_vector, dtype=np.float32).reshape(-1)
    clue = np.asarray(clue_vector, dtype=np.float32).reshape(-1)
    if target.shape != clue.shape:
        raise ValueError("두 임베딩 벡터의 차원이 다릅니다.")
    target_norm = np.linalg.norm(target)
    clue_norm = np.linalg.norm(clue)
    if target_norm == 0 or clue_norm == 0:
        raise ValueError("L2 norm이 0인 벡터는 비교할 수 없습니다.")

    cosine = float(np.dot(target, clue) / (target_norm * clue_norm))
    score = float(np.clip((cosine + 1.0) * 50.0, 0.0, 100.0))
    return SimilarityResult(cosine_similarity=cosine, score=score)
