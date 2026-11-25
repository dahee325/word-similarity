"""Step 3: compare BGE-M3, Word2Vec, and FastText."""

from __future__ import annotations

import html

import streamlit as st

from corpus_loader import load_large_korean_corpus, load_small_corpus
from comparison_service import (
    ModelComparisonResult,
    compare_bge,
    compare_vectors,
    embed_traditional_text,
    train_fasttext,
    train_word2vec,
)
from embedding_service import DEFAULT_MODEL_NAME, load_embedding_model


st.set_page_config(page_title="단어 유사도 모델 비교", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Noto+Sans+KR:wght@400;500;700;800&display=swap');
    :root { --navy:#17213b; --muted:#74809b; --bg:#f4f6fb; --panel:#e9eef8; --card:#fff; --line:#dce3f1; --purple:#6541ed; --purple-soft:#eee9ff; --orange:#c65d05; --teal:#118b83; }
    header[data-testid='stHeader'] { display:block; height:2.25rem; background:transparent; }
    #MainMenu, footer { display:none; }
    /* Keep the expand control available when the browser remembered a collapsed sidebar. */
    button[title='Collapse sidebar'] { display:none!important; }
    [data-testid='collapsedControl'], button[title='Expand sidebar'] { display:flex!important; visibility:visible!important; opacity:1!important; }
    .stApp { background:var(--bg); color:var(--navy); font-family:'Noto Sans KR',sans-serif; }
    .main .block-container { max-width:980px; padding-top:2.8rem!important; padding-bottom:4rem!important; }
    [data-testid='stSidebar'] { background:var(--panel); border-right:1px solid #d9e1f0; }
    [data-testid='stSidebar'] > div:first-child { padding:2.4rem 1.15rem; }
    [data-testid='stSidebar'] h2 { color:var(--navy)!important; font-size:1.15rem!important; margin:0 0 1.5rem; }
    [data-testid='stSidebar'] label { color:var(--navy); font-size:.74rem; }
    [data-testid='stSidebar'] [data-testid='stTextInput'] input { background:white; }
    [data-testid='stSidebar'] .stButton > button { min-height:2.7rem; }
    h1,h2,h3 { color:var(--navy)!important; font-family:'Noto Sans KR',sans-serif; letter-spacing:-.06em; }
    h1 { font-size:2.25rem!important; line-height:1.15!important; margin:.35rem 0 .3rem; }
    h2 { font-size:1.1rem!important; }
    p, .stCaption { color:var(--muted); }
    .page-head { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:1.1rem; }
    .eyebrow { color:#8d78df; font-size:.68rem; font-weight:700; letter-spacing:.08em; }
    .subtitle { color:#62708e; font-size:.82rem; }
    .badge { display:inline-block; background:var(--purple-soft); color:#5940bf; border-radius:999px; padding:.35rem .7rem; font-size:.7rem; margin-top:.55rem; }
    .top-mark { width:54px; height:54px; border-radius:18px; background:var(--purple); color:white; display:grid; place-items:center; font-size:1.5rem; box-shadow:0 8px 18px rgba(101,65,237,.22); }
    .metric-card { background:var(--card); border:1px solid var(--line); border-radius:13px; padding:1rem 1.1rem .8rem; min-height:78px; box-shadow:0 5px 15px rgba(46,66,110,.04); }
    .metric-label { color:#76819b; font-size:.7rem; }
    .metric-value { color:var(--navy); font-size:1.35rem; font-weight:800; margin-top:.18rem; }
    [data-testid='stTextInput'] input { background:#fff; border:1px solid #ccd6e8; border-radius:10px; color:var(--navy); min-height:2.9rem; }
    [data-testid='stTextInput'] input:focus { border-color:var(--purple); box-shadow:0 0 0 1px var(--purple); }
    .stButton > button { background:var(--purple); border:0; border-radius:10px; color:white; font-weight:700; min-height:2.9rem; }
    .stButton > button:hover { background:#5534d1; box-shadow:0 8px 18px rgba(101,65,237,.2); }
    [data-testid='stProgressBar'] > div > div { background:var(--orange); }
    [data-testid='stExpander'] { background:white; border:1px solid var(--line); border-radius:11px; }
    .score-panel { background:linear-gradient(110deg,#fff,#f0f3fb); border:1px solid #d6deed; border-radius:15px; padding:1.15rem 1.3rem 1rem; margin:.85rem 0 .8rem; }
    .score-label { color:#596783; font-size:.76rem; }
    .score-number { color:var(--orange); font-size:3.05rem; font-weight:800; letter-spacing:-.08em; line-height:1.05; }
    .score-state { color:var(--orange); font-size:.8rem; font-weight:700; margin-left:.55rem; }
    .score-meta { color:#7b86a0; font: .68rem 'DM Mono',monospace; }
    .history-title { display:flex; align-items:baseline; gap:.45rem; margin:1rem 0 .55rem; }
    .history-title strong { color:var(--navy); font-size:1rem; }
    .history-title span { color:#7e89a1; font-size:.7rem; }
    .history-item { background:white; border:1px solid var(--line); border-radius:11px; padding:.72rem .8rem; margin:.45rem 0; }
    .history-item.best { border-color:#8062f1; box-shadow:0 4px 12px rgba(101,65,237,.08); }
    .history-top { display:flex; align-items:center; gap:.7rem; }
    .rank { color:#8a94aa; font: .66rem 'DM Mono',monospace; width:25px; }
    .history-name { color:var(--navy); font-size:.78rem; font-weight:700; flex:1; }
    .history-score { color:var(--orange); font-size:.82rem; font-weight:800; }
    .model-note { color:#7d88a0; font-size:.7rem; margin-top:.3rem; }
    .result-help { color:#8b95aa; font-size:.72rem; margin:.4rem 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="BGE-M3 모델을 준비하고 있어요...")
def get_bge_model(model_name: str):
    """Load and cache the BGE-M3 model."""

    return load_embedding_model(model_name)


@st.cache_resource(show_spinner="Word2Vec을 학습하고 있어요...")
def get_word2vec(corpus: tuple[str, ...], vector_size: int, window: int, epochs: int):
    """Train and cache Word2Vec with the selected corpus."""

    return train_word2vec(corpus, vector_size=vector_size, window=window, epochs=epochs)


@st.cache_resource(show_spinner="FastText를 학습하고 있어요...")
def get_fasttext(corpus: tuple[str, ...], vector_size: int, window: int, epochs: int):
    """Train and cache FastText with the selected corpus."""

    return train_fasttext(corpus, vector_size=vector_size, window=window, epochs=epochs)


@st.cache_data(show_spinner="대규모 한국어 말뭉치를 스트리밍으로 준비하고 있어요...")
def get_large_corpus() -> tuple[str, ...]:
    """Cache a bounded Korean Wikipedia sample for local experiments."""

    return tuple(load_large_korean_corpus())


def show_result(result: ModelComparisonResult, rank: int, best: bool = False) -> None:
    """Render one model as a compact history-style result row."""

    similarity = result.similarity
    model_name = html.escape(result.model_name)
    score = similarity.score_0_to_100
    accent = "#c65d05" if result.model_name.startswith("BGE-M3") else "#118b83"
    best_class = " best" if best else ""
    st.markdown(
        f"""
        <div class="history-item{best_class}">
            <div class="history-top"><span class="rank">#{rank:02d}</span><span class="history-name">{model_name}</span><span class="history-score" style="color:{accent}">{score:.2f}</span></div>
            <div style="height:6px;background:#e6ebf4;border-radius:99px;margin:8px 3.1rem 0 2.25rem"><div style="height:6px;width:{max(0,min(100,score)):.2f}%;background:{accent};border-radius:99px"></div></div>
            <div class="model-note">cosine {similarity.cosine_similarity:.6f} · 벡터 차원 {result.first_embedding.dimension}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander(f"{result.model_name} 입력값과 벡터 일부"):
        st.json(
            {
                "first_text": result.first_embedding.text,
                "second_text": result.second_embedding.text,
                "first_dimension": result.first_embedding.dimension,
                "second_dimension": result.second_embedding.dimension,
                "dot_product": similarity.dot_product,
                "first_vector_head": result.first_embedding.vector[:10].tolist(),
                "second_vector_head": result.second_embedding.vector[:10].tolist(),
            }
        )


with st.sidebar:
    st.markdown('<div class="eyebrow">SETTINGS</div>', unsafe_allow_html=True)
    st.header("실험 설정")
    model_choice = st.selectbox(
        "임베딩 모델",
        options=["전체 모델 비교", "BGE-M3", "Word2Vec", "FastText"],
        key="model_choice",
    )
    st.caption("전체 비교 또는 원하는 모델 하나를 선택하세요.")
    corpus_choice = st.selectbox(
        "학습 말뭉치",
        options=["두 말뭉치 비교", "작은 실습 말뭉치", "대규모 한국어 말뭉치"],
        key="corpus_choice",
    )
    vector_size = st.slider("Word2Vec/FastText 벡터 차원", 20, 200, 100, step=20)
    window = st.slider("주변 단어 window", 1, 5, 3)
    epochs = st.slider("학습 반복 횟수", 20, 200, 100, step=20)
    st.divider()
    st.caption("Word2Vec과 FastText가 작은 말뭉치와 Wikipedia 기반 말뭉치에서 어떻게 달라지는지 비교합니다.")
    if st.button("작은 말뭉치 보기", use_container_width=True):
        st.session_state["show_corpus"] = not st.session_state.get("show_corpus", False)


st.markdown('<div class="page-head"><div><div class="eyebrow">EMBEDDING MODEL LAB · STEP 03</div><h1>임베딩 모델 비교</h1><div class="subtitle">같은 단어도 모델에 따라 얼마나 다르게 가까워질까요?</div><span class="badge">BGE-M3 · Word2Vec · FastText</span></div><div class="top-mark">🧪</div></div>', unsafe_allow_html=True)

comparison_results = st.session_state.get("comparison_results", [])
comparison_errors = st.session_state.get("comparison_errors", [])
best_score = max((item.similarity.score_0_to_100 for item in comparison_results), default=None)
metric1, metric2 = st.columns(2)
with metric1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">비교한 모델 수</div><div class="metric-value">{len(comparison_results) if comparison_results else 0}</div></div>', unsafe_allow_html=True)
with metric2:
    best_text = f"{best_score:.2f}" if best_score is not None else "--"
    st.markdown(f'<div class="metric-card"><div class="metric-label">최고 유사도</div><div class="metric-value">{best_text}</div></div>', unsafe_allow_html=True)

input_left, input_right = st.columns([4, 1])
with input_left:
    first_col, second_col = st.columns(2)
    with first_col:
        first_text = st.text_input("첫 번째 단어 또는 짧은 문장", value="사과")
    with second_col:
        second_text = st.text_input("두 번째 단어 또는 짧은 문장", value="과일")
with input_right:
    st.write("")
    st.write("")
    run_comparison = st.button("유사도 계산", type="primary", use_container_width=True)

if st.session_state.get("show_corpus"):
    with st.expander("실습 말뭉치", expanded=True):
        small_corpus = load_small_corpus()
        st.write(f"문장 수: {len(small_corpus)}")
        st.code("\n".join(small_corpus))

if run_comparison:
    results: list[ModelComparisonResult] = []
    errors: list[str] = []
    st.session_state["comparison_results"] = []
    st.session_state["comparison_errors"] = []
    selected_model = st.session_state["model_choice"]
    selected_corpus = st.session_state["corpus_choice"]
    compare_all = selected_model == "전체 모델 비교"
    try:
        if selected_corpus == "두 말뭉치 비교":
            corpus_sets = [("소규모", tuple(load_small_corpus())), ("대규모 Wikipedia", get_large_corpus())]
        elif selected_corpus == "작은 실습 말뭉치":
            corpus_sets = [("소규모", tuple(load_small_corpus()))]
        else:
            corpus_sets = [("대규모 Wikipedia", get_large_corpus())]

        with st.spinner(f"{selected_model} · {selected_corpus} 결과를 계산하고 있어요..."):
            for corpus_label, corpus in corpus_sets:
                if compare_all or selected_model == "BGE-M3":
                    try:
                        bge = get_bge_model(DEFAULT_MODEL_NAME)
                        bge_result = compare_bge(bge, first_text, second_text)
                        results.append(ModelComparisonResult(f"BGE-M3 · {corpus_label}", bge_result.similarity, bge_result.first_embedding, bge_result.second_embedding))
                    except Exception as error:
                        errors.append(f"BGE-M3 · {corpus_label}: {error}")
                if compare_all or selected_model == "Word2Vec":
                    try:
                        word2vec = get_word2vec(corpus, vector_size, window, epochs)
                        results.append(compare_vectors(f"Word2Vec · {corpus_label}", embed_traditional_text(word2vec, first_text), embed_traditional_text(word2vec, second_text)))
                    except Exception as error:
                        errors.append(f"Word2Vec · {corpus_label}: {error}")
                if compare_all or selected_model == "FastText":
                    try:
                        fasttext = get_fasttext(corpus, vector_size, window, epochs)
                        results.append(compare_vectors(f"FastText · {corpus_label}", embed_traditional_text(fasttext, first_text), embed_traditional_text(fasttext, second_text)))
                    except Exception as error:
                        errors.append(f"FastText · {corpus_label}: {error}")
    except Exception as error:
        errors.append(f"말뭉치 준비 중 오류: {error}")

    st.session_state["comparison_results"] = results
    st.session_state["comparison_errors"] = errors
    st.rerun()


if comparison_errors:
    st.warning("일부 모델은 입력 단어가 해당 말뭉치에 없어 계산하지 못했습니다.")
    for comparison_error in comparison_errors:
        st.error(comparison_error)

if comparison_results:
    best_result = max(comparison_results, key=lambda item: item.similarity.score_0_to_100)
    best_score = best_result.similarity.score_0_to_100
    st.markdown('<div class="score-panel">', unsafe_allow_html=True)
    st.markdown(f'<div class="score-label">현재 비교 결과 · {html.escape(best_result.model_name)}</div><div><span class="score-number">{best_score:.2f}</span><span class="score-state">가장 높은 유사도</span></div><div class="score-meta">{html.escape(first_text)} ↔ {html.escape(second_text)} · cosine similarity</div>', unsafe_allow_html=True)
    st.progress(max(0.0, min(1.0, best_score / 100.0)))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="history-title"><strong>모델별 결과</strong><span>높은 유사도 순</span></div>', unsafe_allow_html=True)
    sorted_results = sorted(comparison_results, key=lambda item: item.similarity.score_0_to_100, reverse=True)
    for index, result in enumerate(sorted_results, start=1):
        show_result(result, index, best=index == 1)
else:
    st.markdown('<div class="result-help">단어를 입력하고 유사도 계산을 누르면 모델별 결과가 여기에 표시됩니다.</div>', unsafe_allow_html=True)
