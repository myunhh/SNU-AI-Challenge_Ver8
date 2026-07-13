import sys
from pathlib import Path

# 저장소 루트에서 pytest 실행 시 editable install 없이도 Ver3의 src가 잡히도록
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
