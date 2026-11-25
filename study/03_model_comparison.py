"""3단계: BGE-M3, Word2Vec, FastText 임베딩 결과 비교 화면."""

from __future__ import annotations

import streamlit as st

from comparison_corpus import KOREAN_SENTENCES
from comparison_service import (
    ModelComparisonResult,
    compare_bge,
    compare_vectors,
    embed_traditional_text,
    train_fasttext,
    train_word2vec,
)
from embedding_service import DEFAULT_MODEL_NAME, load_embedding_model


st.set_page_config(page_title="임베딩 모델 비교", page_icon="🔬")


@st.cache_resource(show_spinner="BGE-M3 모델을 준비하는 중입니다...")
def get_bge_model(model_name: str):
    """BGE-M3 모델을 캐시해 반복 로딩을 막는다."""

    return load_embedding_model(model_name)


@st.cache_resource(show_spinner="Word2Vec 모델을 실습 말뭉치로 학습하는 중입니다...")
def get_word2vec(vector_size: int, window: int, epochs: int):
    """설정값별 Word2Vec 모델을 캐시한다."""

    return train_word2vec(KOREAN_SENTENCES, vector_size=vector_size, window=window, epochs=epochs)


@st.cache_resource(show_spinner="FastText 모델을 실습 말뭉치로 학습하는 중입니다...")
def get_fasttext(vector_size: int, window: int, epochs: int):
    """설정값별 FastText 모델을 캐시한다."""

    return train_fasttext(KOREAN_SENTENCES, vector_size=vector_size, window=window, epochs=epochs)


def show_result(result: ModelComparisonResult) -> None:
    """모델별 결과와 실제 입력·벡터 구조를 표시한다."""

    similarity = result.similarity
    with st.container(border=True):
        st.subheader(result.model_name)
        col1, col2 = st.columns(2)
        col1.metric("코사인 유사도", f"{similarity.cosine_similarity:.6f}")
        col2.metric("0~100 표시 점수", f"{similarity.score_0_to_100:.2f}")
        st.progress(similarity.score_0_to_100 / 100.0)
        st.caption(
            f"벡터 차원: {result.first_embedding.dimension} | "
            f"내적: {similarity.dot_product:.6f}"
        )
        with st.expander("입력값과 벡터 데이터 확인"):
            st.json(
                {
                    "first_text": result.first_embedding.text,
                    "second_text": result.second_embedding.text,
                    "first_dimension": result.first_embedding.dimension,
                    "second_dimension": result.second_embedding.dimension,
                    "first_vector_head": result.first_embedding.vector[:10].tolist(),
                    "second_vector_head": result.second_embedding.vector[:10].tolist(),
                }
            )


st.title("3단계: 임베딩 모델 비교")
st.write(
    "같은 단어쌍을 BGE-M3, Word2Vec, FastText로 각각 표현하고 "
    "코사인 유사도 결과가 어떻게 달라지는지 관찰합니다."
)

with st.sidebar:
    st.header("모델 설정")
    model_name = st.text_input("BGE 모델", value=DEFAULT_MODEL_NAME)
    vector_size = st.slider("Word2Vec/FastText 벡터 차원", 20, 200, 100, step=20)
    window = st.slider("주변 단어 window", 1, 5, 3)
    epochs = st.slider("학습 반복 횟수", 20, 200, 100, step=20)
    st.info("Word2Vec과 FastText는 아래의 작은 실습 말뭉치로 직접 학습합니다.")

first_text = st.text_input("첫 번째 단어 또는 짧은 문장", value="사과")
second_text = st.text_input("두 번째 단어 또는 짧은 문장", value="과일")

with st.expander("현재 실습 말뭉치 확인"):
    st.write("문장 수:", len(KOREAN_SENTENCES))
    st.code("\n".join(KOREAN_SENTENCES))

if st.button("세 모델 비교", type="primary"):
    try:
        results: list[ModelComparisonResult] = []
        with st.spinner("세 모델의 임베딩을 계산하고 비교하는 중입니다..."):
            bge = get_bge_model(model_name.strip() or DEFAULT_MODEL_NAME)
            results.append(compare_bge(bge, first_text, second_text))

            word2vec = get_word2vec(vector_size, window, epochs)
            w2v_first = embed_traditional_text(word2vec, first_text)
            w2v_second = embed_traditional_text(word2vec, second_text)
            results.append(compare_vectors("Word2Vec", w2v_first, w2v_second))

            fasttext = get_fasttext(vector_size, window, epochs)
            ft_first = embed_traditional_text(fasttext, first_text)
            ft_second = embed_traditional_text(fasttext, second_text)
            results.append(compare_vectors("FastText", ft_first, ft_second))

        st.session_state["comparison_results"] = results
        st.success("세 모델 비교가 완료되었습니다.")
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error("모델 학습 또는 비교 중 오류가 발생했습니다.")
        st.exception(error)

comparison_results = st.session_state.get("comparison_results", [])
if comparison_results:
    st.subheader("모델별 결과")
    for comparison_result in comparison_results:
        show_result(comparison_result)
