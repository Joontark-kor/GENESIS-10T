# 한국어 벤치마크 측정

학습한 dense 체크포인트를 Hugging Face 형식으로 내보낸 뒤 lm-evaluation-harness로 KoBEST, HAE-RAE, CLIcK, KMMLU를 잰다. 점수와 함께 **오염 검사 결과를 반드시 같이 기록**한다.

## 0. 설치

```bash
pip install -e ".[eval]"
```

이 저장소에서 확인한 버전은 transformers 5.21, lm_eval 0.4.13이다. 로컬 과제로 내보낸 모델을 lm_eval에 넣어 끝까지 도는 것까지 확인했고, 실제 벤치마크 데이터셋 다운로드는 Hugging Face 접속이 되는 환경에서 해야 한다.

## 1. 내보내기

```bash
python -m genesis.export_hf --checkpoint checkpoints/dense-100m/model.pt --output exports/dense-100m-hf
```

마지막 줄에 `verified: max |logit difference| = ...`가 나오면 원본과 내보낸 모델의 출력이 일치한 것이다. MoE 체크포인트와 바이트 토크나이저는 아직 내보낼 수 없다.

## 2. 오염 검사 (점수를 재기 전에)

```bash
python -m genesis.bench_to_jsonl --benchmarks kobest haerae click kmmlu --output data/bench
for b in kobest haerae click kmmlu; do
  python -m genesis.contamination --train data/clean/v1 --benchmark data/bench/$b.jsonl --fields text \
      --report reports/contamination_$b.json --blocklist reports/blocklist.txt
done
```

`contaminated_items`가 0이 아니면 해당 문서를 빼고 다시 정제·학습하거나, 최소한 점수 옆에 오염률을 적는다.

```bash
python -m genesis.data.clean --sources configs/sources.json --output data/clean/v2 --exclude-sha256 reports/blocklist.txt
```

## 3. 측정

```bash
lm_eval --model hf --model_args pretrained=exports/dense-100m-hf,dtype=bfloat16 \
    --tasks kobest,haerae,click --batch_size 16 --device cuda:0 \
    --output_path reports/lm_eval/dense-100m --log_samples
```

KMMLU는 45개 과목이라 오래 걸리고, 1B 이하 모델은 찍기 수준(25%) 근처가 정상이다. 100M 단계에서는 KoBEST·HAE-RAE·CLIcK로 진척을 보고, KMMLU는 1B부터 잰다.

| 벤치마크 | lm_eval 과제 이름 | 형식 | 무작위 기준 |
|---|---|---|---|
| KoBEST | `kobest` (boolq, copa, hellaswag, sentineg, wic) | 선택지 로그우도 | 과제별 50% 또는 25% |
| HAE-RAE | `haerae` (5개 하위 과제) | 5지선다 | 20% |
| CLIcK | `click` | 4~5지선다 | 20~25% |
| KMMLU | `kmmlu` (45개 과목) | 4지선다 | 25% |

## 4. 기록

- 모델(체크포인트 경로, 스텝, 학습 토큰), 토크나이저 해시, 내보내기 검증값
- 과제별 `acc`, `acc_norm`과 표준오차
- 과제별 오염률(`reports/contamination_*.json`)
- 평가 설정: few-shot 수(기본 0), 배치, dtype, lm_eval 버전

같은 설정 파일과 버전으로 재야 다른 모델과 비교할 수 있다. 공개 리더보드 점수와 비교할 때는 그쪽의 프롬프트·few-shot 설정이 같은지 먼저 확인한다.
