# 학습용 임베딩 화면: 항상 8501 포트를 사용한다.
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$script = Join-Path $projectRoot 'study\02_similarity.py'

& $python -m streamlit run $script --server.address localhost --server.port 8501
