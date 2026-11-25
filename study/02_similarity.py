"""2단계: 두 텍스트의 BGE-M3 임베딩과 코사인 유사도를 확인하는 화면."""

from __future__ import annotations

import streamlit as st

from embedding_service import DEFAULT_MODEL_NAME, embed_text, load_embedding_model
from similarity_service import SimilarityResult, calculate_cosine_similarity


st.set_page_config(page_title="코사인 유사도 학습", page_icon="📐")


@st.cache_resource(show_spinner="BGE-M3 모델을 처음 한 번만 로딩하는 중입니다...")
def get_model(model_name: str):
    """모델을 Streamlit 리소스 캐시에 저장해 반복 로딩을 막는다."""

    return load_embedding_model(model_name)


def show_result(result: SimilarityResult) -> None:
    """입력값·중간 계산값·최종 반환값을 화면에 표시한다."""

    st.subheader("계산 결과")
    st.metric("코사인 유사도", f"{result.cosine_similarity:.6f}")
    st.progress(result.score_0_to_100 / 100.0)
    st.metric("0~100 표시 점수", f"{result.score_0_to_100:.2f}")

    with st.expander("벡터 연산 중간 결과 확인", expanded=True):
        st.write("내적 (A · B)", result.dot_product)
        st.write("첫 번째 벡터 L2 norm (||A||)", result.first_norm)
        st.write("두 번째 벡터 L2 norm (||B||)", result.second_norm)
        st.code(
            "cosine(A, B) = dot(A, B) / (norm(A) * norm(B))\n"
            f"           = {result.dot_product:.6f} / "
            f"({result.first_norm:.6f} * {result.second_norm:.6f})\n"
            f"           = {result.cosine_similarity:.6f}"
        )

    with st.expander("반환 데이터 구조 확인"):
        st.json(
            {
                "first_text": result.first_text,
                "second_text": result.second_text,
                "cosine_similarity": result.cosine_similarity,
                "score_0_to_100": result.score_0_to_100,
                "first_dimension": result.first_dimension,
                "second_dimension": result.second_dimension,
            }
        )


st.title("2단계: 코사인 유사도")
st.write(
    "두 텍스트를 각각 BGE-M3 벡터로 변환한 다음, 두 벡터의 방향이 얼마나 가까운지 계산합니다."
)

with st.sidebar:
    st.header("모델 설정")
    model_name = st.text_input("임베딩 모델", value=DEFAULT_MODEL_NAME)
    st.info("모델은 한 번 로딩되면 캐시되어 반복 사용됩니다.")

first_text = st.text_input("첫 번째 텍스트", value="사과")
second_text = st.text_input("두 번째 텍스트", value="과일")

if st.button("코사인 유사도 계산", type="primary"):
    try:
        with st.spinner("두 텍스트를 임베딩하고 벡터를 비교하는 중입니다..."):
            model = get_model(model_name.strip() or DEFAULT_MODEL_NAME)
            first_embedding = embed_text(model, first_text)
            second_embedding = embed_text(model, second_text)
            result = calculate_cosine_similarity(first_embedding, second_embedding)
        st.session_state["last_similarity_result"] = result
        st.success("계산이 완료되었습니다.")
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error("모델 로딩 또는 유사도 계산 중 오류가 발생했습니다.")
        st.exception(error)

last_result = st.session_state.get("last_similarity_result")
if isinstance(last_result, SimilarityResult):
    show_result(last_result)
