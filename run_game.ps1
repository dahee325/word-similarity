# 게임 화면: 항상 8502 포트를 사용한다.
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$script = Join-Path $projectRoot 'game\app.py'

& $python -m streamlit run $script --server.address localhost --server.port 8502
