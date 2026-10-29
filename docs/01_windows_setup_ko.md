# Windows에서 GENESIS-10T 실행하기

## 0. 준비

Python 3.10 이상 설치. VS Code 권장. 설치 시 `Add Python to PATH`를 선택한다. PowerShell이나 VS Code 터미널에서 프로젝트 폴더를 연다.

```powershell
cd GENESIS-10T
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 1. 코드 정확성 검사

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

합격 기준: 자동 테스트 전부 통과.

## 2. 사용 허가받은 학습 데이터 준비

`data/raw/sample.txt`는 프로젝트를 시험할 목적으로 직접 작성한 데모 텍스트다. 외부 데이터로 교체할 때는 학습 사용권, 라이선스, 개인정보를 먼저 확인한다.

```powershell
.\.venv\Scripts\python.exe -m genesis.prepare --input data/raw --output data/processed --source self-authored-demo --license self-authored
```

합격 기준: `data/processed`에 `train_00000.bin`, `val_00000.bin`, `test_00000.bin`, `manifest.json` 생성. 샘플은 문서가 1개뿐이라 val/test를 토큰 단위로 잘라내며, manifest의 `split_strategy`에 그 사실이 기록된다.

## 3. 사전학습과 텍스트 생성

```powershell
.\.venv\Scripts\python.exe -m genesis.train --config configs/debug.json --data data/processed --out-dir checkpoints/debug --steps 25 --batch-size 4 --eval-every 5
.\.venv\Scripts\python.exe -m genesis.generate --checkpoint checkpoints/debug/model.pt --prompt "인공지능은" --tokens 60
```

합격 기준: 학습 로그에 loss와 val_bpb 출력, `checkpoints/debug/model.pt`(추론용)와 `ckpt_last.pt`(재개용) 생성, 생성 명령 오류 없음. **샘플 데이터가 너무 작아 의미 있는 답변이 나올 것으로 기대하면 안 된다.**

## 4. 10조 가설 모델의 크기만 계산

```powershell
.\.venv\Scripts\python.exe -m genesis.estimate --hypothetical-10t
```

이 명령은 설계한 가설 모델의 파라미터 수만 계산하며, 10T를 메모리에 할당하거나 학습하지 않는다.

## 5. 그래픽카드 사용

현재 기본 경로는 CPU 디버깅을 지원한다. NVIDIA CUDA GPU를 활용하려면 정확한 드라이버 및 GPU 종류에 맞춰 PyTorch 설치 명령을 공식 사이트에서 확인한다: https://pytorch.org/get-started/locally/ . 향후 다중 GPU/MoE 분산 실험은 WSL2 또는 리눅스에서 진행하는 것을 권장한다.

## 6. 실제 데이터로 넘어가기

실제 한국어 데이터 파이프라인은 [02_pipeline_ko.md](02_pipeline_ko.md), 다음 할 일은 [03_next_steps_ko.md](03_next_steps_ko.md)를 본다.

## 7. GitHub에 독립 저장소 올리기

GitHub에서 새 **빈** 저장소 `GENESIS-10T`를 만들고, 로컬에서 프로젝트 폴더를 열어 다음 명령을 실행한다. 저장소에 README/.gitignore/License 파일을 GitHub에서 추가로 생성하지 말 것(이미 프로젝트에 있음).

```powershell
git init
git add .
git commit -m "Initialize GENESIS-10T research baseline"
git branch -M main
git remote add origin https://github.com/<본인-GitHub-아이디>/GENESIS-10T.git
git push -u origin main
```

본인이 실제 GitHub 계정의 사용자 이름으로 `<본인-GitHub-아이디>`를 바꾼다. GitHub 인증이 요구될 수 있다. 기존 `Code` 저장소는 이 과정에서 변경되지 않는다.
