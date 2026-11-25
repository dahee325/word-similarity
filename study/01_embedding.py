"""1단계: Streamlit에서 BGE-M3 단어 임베딩을 관찰하는 실습 화면.

실행:
    streamlit run study/01_embedding.py
"""

from __future__ import annotations

import streamlit as st

from embedding_service import (
    DEFAULT_MODEL_NAME,
    EmbeddingResult,
    embed_text,
    load_embedding_model,
)


st.set_page_config(page_title="한국어 단어 임베딩 실습", page_icon="🧠")


@st.cache_resource(show_spinner="BGE-M3 모델을 처음 한 번만 로딩하는 중입니다...")
def get_model(model_name: str):
    """모델을 Streamlit 리소스 캐시에 저장해 재사용한다."""

    return load_embedding_model(model_name)


def show_embedding_result(result: EmbeddingResult) -> None:
    """임베딩 결과의 입력값·중간 결과·반환값을 화면에 표시한다."""

    st.subheader("임베딩 결과")
    st.write("입력값", result.text)
    col1, col2 = st.columns(2)
    col1.metric("벡터 차원", f"{result.dimension:,}")
    col2.metric("벡터 L2 norm", f"{result.norm:.6f}")

    with st.expander("반환된 벡터 일부 확인"):
        st.caption("전체 벡터 대신 앞 20개 값을 표시합니다.")
        st.code(result.vector[:20].tolist())

    with st.expander("전체 반환 데이터 구조 확인"):
        st.json(
            {
                "text": result.text,
                "vector": "numpy.ndarray (전체 값은 위에서 일부 확인)",
                "dimension": result.dimension,
                "norm": result.norm,
            }
        )


st.title("🧠 한국어 단어 유사도 게임 - 1단계")
st.write(
    "이번 단계에서는 BGE-M3가 한국어 단어를 숫자 벡터로 바꾸는 과정을 확인합니다. "
    "유사도 점수 계산은 다음 단계에서 추가합니다."
)

with st.sidebar:
    st.header("모델 설정")
    model_name = st.text_input("임베딩 모델", value=DEFAULT_MODEL_NAME)
    normalize_embeddings = st.checkbox(
        "임베딩 벡터 L2 정규화",
        value=False,
        help="다음 단계에서 코사인 유사도를 계산할 때 활용할 수 있습니다.",
    )
    st.info("모델은 캐시되므로 입력을 바꾸거나 버튼을 눌러도 매번 다시 로딩되지 않습니다.")

st.subheader("단어 입력")
word = st.text_input("임베딩할 한국어 단어 또는 짧은 문장", value="사과")

if st.button("임베딩 계산", type="primary"):
    try:
        with st.spinner("텍스트를 벡터로 변환하는 중입니다..."):
            model = get_model(model_name.strip() or DEFAULT_MODEL_NAME)
            result = embed_text(
                model,
                word,
                normalize_embeddings=normalize_embeddings,
            )
        st.session_state["last_embedding_result"] = result
        st.success("임베딩 계산이 완료되었습니다.")
    except ValueError as error:
        st.warning(str(error))
    except Exception as error:
        st.error("모델 로딩 또는 임베딩 중 오류가 발생했습니다.")
        st.exception(error)

last_result = st.session_state.get("last_embedding_result")
if isinstance(last_result, EmbeddingResult):
    show_embedding_result(last_result)
