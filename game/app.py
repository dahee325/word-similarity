"""BGE-M3 기반 한국어 단어 유사도 추측 게임의 Streamlit 화면."""

from __future__ import annotations

import random
from typing import Any

import streamlit as st

from embedding_service import DEFAULT_MODEL_NAME, embed_text, load_embedding_model
from similarity import SimilarityResult, calculate_cosine_similarity
from word_bank import WORD_BANK


st.set_page_config(page_title="Word Signal", page_icon="W", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#17202a; --muted:#6d747c; --line:#e8e4dd; --paper:#fbfaf7; --accent:#ff694a; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#f2efe9; border-right:1px solid var(--line); }
    .block-container { max-width:1180px; padding-top:2.5rem; padding-bottom:4rem; }
    h1,h2,h3,p,button,input { font-family:'Space Grotesk', sans-serif; }
    h1 { letter-spacing:-.055em; font-size:clamp(2.3rem,5vw,4.6rem)!important; line-height:.95!important; margin-bottom:.5rem!important; white-space:pre-line; }
    h2 { letter-spacing:-.035em; }
    .brand { display:flex; justify-content:space-between; align-items:center; margin-bottom:3rem; border-bottom:1px solid var(--line); padding-bottom:1rem; }
    .brand-name,.brand-status,.score-label,.score-note,.section-label,.history-score { font-family:'DM Mono',monospace; }
    .brand-name { font-size:.78rem; letter-spacing:.12em; font-weight:500; }
    .brand-status { color:var(--accent); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; }
    .eyebrow,.section-label { color:var(--accent); font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; }
    .lede { max-width:620px; color:var(--muted); font-size:1.03rem; line-height:1.65; margin-bottom:2.3rem; }
    .target-card { background:var(--ink); color:white; border-radius:18px; padding:1.45rem 1.6rem; margin:1rem 0 1.3rem; }
    .target-label { color:#aeb7bd; font-family:'DM Mono',monospace; font-size:.67rem; letter-spacing:.1em; text-transform:uppercase; }
    .target-word { font-size:2.3rem; font-weight:600; letter-spacing:-.05em; margin-top:.35rem; }
    .score-card { background:white; border:1px solid var(--line); border-radius:18px; padding:1.4rem 1.6rem; margin:1rem 0; }
    .score-label { color:var(--muted); font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
    .score-value { color:var(--accent); font-size:3.5rem; font-weight:700; letter-spacing:-.07em; line-height:1.1; }
    .score-note { color:var(--muted); font-size:.72rem; margin-top:.35rem; }
    .section-label { color:var(--muted); margin:1.8rem 0 .65rem; }
    .history-row { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding:.75rem 0; }
    .history-word { font-weight:600; }
    .history-score { color:var(--accent); font-size:.82rem; }
    .hint { color:var(--muted); font-size:.88rem; }
    [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; }
    [data-testid="stMetricLabel"] { font-family:'DM Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }
    .stButton > button { border-radius:10px; border:1px solid var(--ink); font-weight:600; }
    div[data-testid="stButton"] button[kind="primary"] { background:var(--accent); border-color:var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="모델을 준비하는 중입니다...")
def get_model(model_name: str):
    """모델을 Streamlit 리소스 캐시에 저장해 반복 로딩을 막는다."""

    return load_embedding_model(model_name)


def initialize_game() -> None:
    """무작위 제시어를 뽑고 현재 게임의 시도 이력을 초기화한다."""

    st.session_state.target_word = random.choice(WORD_BANK)
    st.session_state.history = []
    st.session_state.pop("last_result", None)


def reset_game() -> None:
    """제시어와 시도 이력을 삭제해 새 게임 대기 상태로 만든다."""

    st.session_state.pop("target_word", None)
    st.session_state.history = []
    st.session_state.pop("last_result", None)


def render_score(result: SimilarityResult) -> None:
    """최근 유사도 점수와 실제 코사인 값을 카드로 표시한다."""

    st.markdown(
        f"""<div class="score-card"><div class="score-label">Similarity score</div>
        <div class="score-value">{result.score:.2f}<span style="font-size:1.25rem;color:#6d747c"> / 100</span></div>
        <div class="score-note">raw cosine &nbsp; {result.cosine_similarity:.6f}</div></div>""",
        unsafe_allow_html=True,
    )
    st.progress(result.score / 100.0)


def render_history(history: list[dict[str, Any]]) -> None:
    """시도 이력을 점수 내림차순으로 정렬해 표시한다."""

    st.markdown('<div class="section-label">Attempts · high to low</div>', unsafe_allow_html=True)
    if not history:
        st.markdown('<div class="hint">아직 입력된 단어가 없습니다.</div>', unsafe_allow_html=True)
        return

    sorted_history = sorted(history, key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(sorted_history, start=1):
        st.markdown(
            f"""<div class="history-row"><span><span class="hint">{rank:02d}</span>&nbsp;&nbsp;
            <span class="history-word">{item['clue_word']}</span></span>
            <span class="history-score">{item['score']:.2f}</span></div>""",
            unsafe_allow_html=True,
        )


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("## WORD SIGNAL")
    st.caption("A word. A hunch. One hidden answer.")
    view_mode = st.radio("화면 역할", ("A · 제시어 확인", "참가자 · 제시어 추측"))
    model_name = st.text_input("Embedding model", value=DEFAULT_MODEL_NAME)
    st.divider()
    if st.button("새 제시어 뽑기", use_container_width=True):
        initialize_game()
        st.rerun()
    if st.button("게임 초기화", use_container_width=True):
        reset_game()
        st.rerun()
    st.caption("제시어는 이 브라우저 세션 안에서만 유지됩니다.")

st.markdown(
    '<div class="brand"><span class="brand-name">WORD SIGNAL / 01</span>'
    '<span class="brand-status">live · local game</span></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">Korean word similarity game</div>', unsafe_allow_html=True)
st.title("한 단어로,\n제시어를 맞혀보세요.")
st.markdown(
    '<div class="lede">A는 숨겨진 제시어와 연결되는 단어를 던집니다. '
    '참가자는 공개된 점수의 흐름만 보고 제시어를 추리합니다.</div>',
    unsafe_allow_html=True,
)

if "target_word" not in st.session_state:
    st.info("사이드바에서 ‘새 제시어 뽑기’를 눌러 라운드를 시작하세요.")
    st.stop()

target_word = st.session_state.target_word
history = st.session_state.history

left, right = st.columns([1.35, 1], gap="large")
with left:
    if view_mode == "A · 제시어 확인":
        st.markdown(
            f'<div class="target-card"><div class="target-label">A only · hidden prompt</div>'
            f'<div class="target-word">{target_word}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Drop a clue</div>', unsafe_allow_html=True)
        clue_word = st.text_input("제시어와 관련된 단어", key="clue_word", label_visibility="collapsed", placeholder="관련 단어를 입력하세요")
        if st.button("유사도 확인", type="primary", use_container_width=True):
            try:
                with st.spinner("두 단어의 의미 공간을 비교하는 중입니다..."):
                    model = get_model(model_name.strip() or DEFAULT_MODEL_NAME)
                    result = calculate_cosine_similarity(
                        embed_text(model, target_word),
                        embed_text(model, clue_word),
                    )
                history.append({"clue_word": clue_word.strip(), "score": result.score, "cosine_similarity": result.cosine_similarity})
                st.session_state.last_result = result
                st.rerun()
            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error("모델 로딩 또는 유사도 계산 중 오류가 발생했습니다.")
                st.exception(error)
    else:
        st.markdown('<div class="target-card"><div class="target-label">Guessing mode</div><div class="target-word">제시어는 비공개</div></div>', unsafe_allow_html=True)
        st.markdown('<p class="hint">A가 던진 단어와 점수를 보고 다음 제시어를 추리하세요.</p>', unsafe_allow_html=True)

with right:
    last_result = st.session_state.get("last_result")
    if isinstance(last_result, SimilarityResult):
        render_score(last_result)
    else:
        st.markdown('<div class="score-card"><div class="score-label">Waiting for the first clue</div><div class="score-value">—</div><div class="score-note">점수가 여기에 표시됩니다</div></div>', unsafe_allow_html=True)
    attempt_count = len(history)
    best_score = max((item["score"] for item in history), default=0.0)
    metric_left, metric_right = st.columns(2)
    metric_left.metric("시도 횟수", attempt_count)
    metric_right.metric("최고 점수", f"{best_score:.2f}")

render_history(history)
