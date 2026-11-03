# GENESIS-Applied: 공개 모델을 한국 기업 재무 분석에 특화하기

**목표:** 공개 LLM을 LoRA/QLoRA로 학습해, 한국 상장사의 공시(OpenDART) 재무제표에서 숫자를 정확히 읽고 재무비율을 계산·검증하는 모델을 만든다. 일본(EDINET)은 한국 결과가 나온 뒤의 선택 확장이다. 릴리스마다 **정확도·비용·속도**를 같은 형식으로 공개한다.

정답은 사람이 채점하지 않고 표의 숫자로 계산해 만든다. 그래서 "몇 % 맞혔다"를 누구나 다시 확인할 수 있다. 투자 권유가 아니라 분석 보조로 설계한다.

```
OpenDART ──fetch──▶ records.jsonl ──build──▶ sft_train / sft_val (학습)
                                          └─▶ eval_val / eval_test (평가, 학습에 없는 회사)
기반 모델 ──평가──▶ base.json        기반 모델 + QLoRA ──평가──▶ fin-v1.json ──compare──▶ 비교표
```

## 1차 과제 (fin-v1)

| 종류 | 예시 질문 | 정답 계산 |
|---|---|---|
| extract | 2025년 자산총계는? | 표의 값 그대로 |
| operating_margin | 영업이익률 | 영업이익 ÷ 매출액 × 100 |
| net_margin | 순이익률 | 당기순이익 ÷ 매출액 × 100 |
| debt_ratio | 부채비율 | 부채총계 ÷ 자본총계 × 100 |
| roe | ROE(기말 자본 기준) | 당기순이익 ÷ 기말 자본총계 × 100 |
| equity_ratio | 자기자본비율 | 자본총계 ÷ 자산총계 × 100 |
| revenue_growth | 매출액 증가율 | (당해 − 전년) ÷ 전년 × 100 |

- 금액은 백만 단위로 보여 주고, 정답은 **보여 준 숫자로** 계산한다(소수 둘째 자리, 허용 오차 0.01%p).
- 회사 단위로 train/val/test를 나눈다. 평가 회사는 학습에 한 번도 나오지 않는다.
- 일본어 질문 템플릿(`--langs ko ja`)은 코드에 남아 있지만, 지금은 한국어(`ko`)만 쓴다.

## 0. 준비 (집 PC, 한 번만)

1. OpenDART 인증키 받기: https://opendart.fss.or.kr → 회원가입 → 인증키 신청(무료, 40자).
2. 키를 환경변수로 저장. 코드나 저장소에는 절대 넣지 않는다.
   ```bash
   echo 'export DART_API_KEY="발급받은_40자_키"' >> ~/.bashrc && source ~/.bashrc
   ```
3. 패키지 설치
   ```bash
   cd ~/work/GENESIS-10T && source .venv/bin/activate && git checkout main && git pull
   pip install -e ".[applied]"
   ```

## 1. 데이터 수집과 과제 만들기 (집 PC, CPU로 충분)

```bash
python -m genesis.applied.build_fin_tasks fetch --years 2023 2024 2025 --companies 300 --out data/applied/fin_v1
python -m genesis.applied.build_fin_tasks build --out data/applied/fin_v1 --langs ko
cat data/applied/fin_v1/manifest.json
```

- 상장사 300곳 × 3년이면 API 호출은 1,000~2,000회 정도다. OpenDART의 일일 한도(약 2만 회) 안이다.
- 응답은 `data/applied/cache/dart/`에 저장되므로 다시 실행해도 호출하지 않는다. 한도 초과(020)가 나면 진행분을 저장하고 멈추며, 다음 날 같은 명령으로 이어서 받는다.
- `manifest.json`의 `tasks`에서 split별 과제 수를 확인한다.

## 2. 기반 모델 고르기

조건: (1) 상업 사용 가능한 라이선스, (2) 한국어를 잘함, (3) 8B 이하라 QLoRA로 GPU 1장에서 학습 가능.

- 먼저 작은 모델(1~4B)로 전체 흐름을 싸게 확인하고, 그다음 8B급으로 올린다.
- 예: Qwen3 계열(Apache 2.0으로 공개됨). Qwen3는 기본으로 "생각" 블록을 쓰므로 학습·평가 모두 `--template-kwargs '{"enable_thinking": false}'`를 준다.
- 한국어 특화 공개 모델도 후보다. **모델 카드의 라이선스를 직접 확인**하고 `run.json`의 기록과 함께 남긴다.

## 3. GPU에서: 기준선 → 학습 → 비교

QLoRA 8B는 24GB급 GPU 1장(L4, A10G, RTX 4090 등)이면 된다. 클라우드에서 Ubuntu GPU 서버를 빌려 저장소를 받고 `pip install -e ".[applied,qlora]"`, 그리고 `data/applied/fin_v1`을 `rsync`로 보낸다(런북 8단계와 같은 방법).

```bash
MODEL=Qwen/Qwen3-4B
TK='{"enable_thinking": false}'
# (1) 학습 전 기준선
python -m genesis.applied.evaluate_tasks run --model $MODEL --template-kwargs "$TK" \
    --data data/applied/fin_v1/eval_test.jsonl --name base --gpu-hour-price <시간당가격> \
    --out reports/applied/base.json
# (2) QLoRA 학습
python -m genesis.applied.finetune --base-model $MODEL --template-kwargs "$TK" --qlora \
    --train data/applied/fin_v1/sft_train.jsonl --val data/applied/fin_v1/sft_val.jsonl \
    --output runs/applied/fin-v1 --epochs 2 --max-length 1024
# (3) 학습 후 평가와 비교
python -m genesis.applied.evaluate_tasks run --model $MODEL --adapter runs/applied/fin-v1/adapter \
    --template-kwargs "$TK" --data data/applied/fin_v1/eval_test.jsonl --name fin-v1 \
    --gpu-hour-price <시간당가격> --out reports/applied/fin-v1.json
python -m genesis.applied.evaluate_tasks compare reports/applied/base.json reports/applied/fin-v1.json
```

공유할 것: `reports/applied/*.json`(용량이 크면 마지막 비교표만), `runs/applied/fin-v1/run.json`, `data/applied/fin_v1/manifest.json`.

## 확인한 것과 아직 아닌 것

- 확인함(자동 테스트): DART 응답 정규화, 한도 초과 시 진행분 보존·재개, 과제 계산, 회사 단위 분할, 응답 토큰만 학습하는 라벨링, 소형 모델로 LoRA 학습 → 저장 → 평가 → 비교 전체 흐름(CPU).
- 아직 아님: 실제 OpenDART 호출(키가 필요), 실제 기반 모델 다운로드와 GPU QLoRA 학습.

## 다음 단계 후보 (한국 집중)

1. 과제 심화: 여러 해 추세, 연결과 별도 재무제표 차이, 같은 업종 기업 비교, 공시 본문 문장 속 숫자 찾기
2. 공개 벤치마크 병행: KMMLU 회계·세무·경제·경영·부동산 과목, KRX-Bench
3. Hugging Face에 어댑터, 모델 카드, 평가 보고서 공개
4. (선택, 나중에) 일본 EDINET 수집과 일본어 과제
