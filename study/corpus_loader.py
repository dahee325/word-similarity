"""Load the small practice corpus and a streamed Korean Wikipedia corpus."""

from __future__ import annotations

import re

from comparison_corpus import KOREAN_SENTENCES


ARTICLE_SENTENCE_SPLITTER = re.compile(r"[\r\n]+|(?<=[.!?。！？])\s*")


def load_small_corpus() -> list[str]:
    """Return the 31 hand-written sentences used for the first experiments."""

    return list(KOREAN_SENTENCES)


def load_large_korean_corpus(
    *,
    max_articles: int = 2_000,
    max_sentences: int = 50_000,
) -> list[str]:
    """Stream cleaned Korean Wikipedia articles and split them into sentences.

    The dataset is intentionally capped so the learning UI remains runnable on a
    local machine. It is still much larger and more varied than the 31-sentence
    practice corpus.
    """

    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.ko",
        split="train",
        streaming=True,
    )

    sentences: list[str] = []
    for article_index, article in enumerate(dataset):
        if article_index >= max_articles or len(sentences) >= max_sentences:
            break

        text = str(article.get("text", "")).strip()
        for sentence in ARTICLE_SENTENCE_SPLITTER.split(text):
            normalized = " ".join(sentence.split())
            if len(normalized) >= 10:
                sentences.append(normalized)
                if len(sentences) >= max_sentences:
                    break

    if not sentences:
        raise RuntimeError("한국어 Wikipedia 말뭉치를 불러오지 못했습니다.")
    return sentences
