import sys
from pathlib import Path

# src 레이아웃을 설치 없이 import 가능하게 (3090/VESSL에서도 동일하게 동작)
sys.path.insert(0, str(Path(__file__).parent / "src"))
