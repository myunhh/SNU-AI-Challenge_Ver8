#실행 방법
#아래 두 명령어를 순차 실행
#실행 data디렉터리에
  #data\train, data\test 폴더에 데이터와
  #csv 파일들이 다운로드됨

#export KAGGLE_API_TOKEN=
#python data_download.py

import os
import shutil
from pathlib import Path

import kagglehub

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
REQUIRED_ITEMS = ["train.csv", "test.csv", "train", "test"]
OPTIONAL_ITEMS = ["sample_submission.csv"]


def _find_data_root(download_path: Path) -> Path:
    """kagglehub가 반환한 경로 밑에서 train.csv가 실제로 있는 폴더를 찾는다
    (예: .../snuaichallenge/snuaichallenge_data/train.csv 처럼 한 번 더 감싸진 경우까지 처리)."""
    if (download_path / "train.csv").exists():
        return download_path
    for child in download_path.iterdir():
        if child.is_dir() and (child / "train.csv").exists():
            return child
    matches = list(download_path.rglob("train.csv"))
    if not matches:
        raise FileNotFoundError(f"train.csv를 {download_path} 아래에서 찾지 못했습니다.")
    return matches[0].parent


# 인증: KAGGLE_API_TOKEN 환경변수가 있으면 kagglehub가 자동으로 읽어서 무인 다운로드된다
# (토큰은 절대 이 파일에 하드코딩하지 말 것 — 이 저장소는 public GitHub에 올라가 있음).
#   실행 전:  export KAGGLE_API_TOKEN=발급받은토큰
# 미설정 시에만 대화형 프롬프트로 폴백.
if not os.environ.get("KAGGLE_API_TOKEN"):
    kagglehub.login()

# 다운로드 실행
raw_path = Path(kagglehub.competition_download("snuaichallenge"))
print("kagglehub 다운로드 경로:", raw_path)

src_root = _find_data_root(raw_path)
print("실제 데이터 폴더:", src_root)

DATA_DIR.mkdir(exist_ok=True)
for name in REQUIRED_ITEMS + OPTIONAL_ITEMS:
    src = src_root / name
    if not src.exists():
        if name in OPTIONAL_ITEMS:
            continue
        raise FileNotFoundError(f"{src} 를 찾을 수 없습니다.")
    dest = DATA_DIR / name
    if dest.exists():
        print(f"이미 존재해서 건너뜀: {dest}")
        continue
    shutil.move(str(src), str(dest))
    print(f"이동: {src} -> {dest}")

# kagglehub 캐시에 남은 빈 래퍼 폴더 정리 (다음 실행 시 정상적으로 재다운로드되도록)
shutil.rmtree(raw_path, ignore_errors=True)

print("완료:", sorted(p.name for p in DATA_DIR.iterdir()))
