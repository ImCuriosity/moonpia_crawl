#!/usr/bin/env bash
# ============================================================
#  문피아 데이터 수집기 - macOS / Linux 실행 스크립트
#    chmod +x run.sh && ./run.sh
# ============================================================
set -u

cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

echo
echo "============================================================"
echo "  문피아 웹소설 데이터 수집기"
echo "============================================================"
echo

# ---- 1. Python 찾기 ----------------------------------------
PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)' 2>/dev/null; then
            PY="$cand"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "  [오류] Python 3.8 이상이 필요합니다."
    echo
    echo "    macOS : brew install python3"
    echo "    Ubuntu: sudo apt install python3 python3-venv"
    echo
    exit 1
fi

echo "  Python: $($PY --version 2>&1)"

# ---- 2. 가상환경 준비 --------------------------------------
if [ ! -d ".venv" ]; then
    echo "  처음 실행이라 준비 작업을 합니다. 1~2분 걸립니다..."
    echo
    # </dev/null 로 준비 단계가 사용자 입력을 삼키지 않게 한다
    echo "  [1/2] 가상환경 만드는 중..."
    if ! "$PY" -m venv .venv </dev/null; then
        echo "  [오류] 가상환경 생성 실패."
        echo "         Ubuntu라면: sudo apt install python3-venv"
        exit 1
    fi
    echo "  [2/2] 필요한 패키지 설치 중... (requests, pandas)"
    .venv/bin/python -m pip install --upgrade pip --quiet </dev/null
    if ! .venv/bin/python -m pip install --quiet requests pandas </dev/null; then
        echo "  [오류] 패키지 설치 실패. 인터넷 연결을 확인해 주세요."
        exit 1
    fi
    echo "  준비 완료!"
fi

# ---- 3. 실행 -----------------------------------------------
.venv/bin/python -m munpia.wizard
EXITCODE=$?

echo
echo "============================================================"
if [ $EXITCODE -ne 0 ]; then
    echo "  오류로 종료되었습니다 (코드 $EXITCODE)."
fi
echo "============================================================"
exit $EXITCODE
