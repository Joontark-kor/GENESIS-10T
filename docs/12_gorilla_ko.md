# Gorilla — 모델은 식을 쓰고, 계산은 계산기가 한다

## 왜 만들었나

fin-v1 첫 QLoRA(2026-11-05): 전체 정확도 85.3% → 84.0%, 개선 없음. 틀린 답을 읽어 보니 수식은 맞게 쓰는데
나눗셈과 반올림이 틀렸다(정답 −4.84, 모델 −4.88). 숫자 추출은 100%였다. 병목은 지식이 아니라
**모델의 직접 산술**이고, 파인튜닝으로는 정밀 계산이 고쳐지지 않는다.

## 구조 (가장 작은 형태)

```
질문 ──▶ 모델: "계산식: (a - b) / b * 100" 한 줄 ──▶ genesis.tools.calculate ──▶ "정답: -4.84"
```

- `genesis/tools.py`: 사칙연산·거듭제곱만 허용하는 안전한 계산기(이름·함수 호출·속성 불가, 지수 12 제한).
  천 단위 쉼표, `× ÷ −`, 끝의 `= 결과`는 정리해서 받는다.
- `genesis/harness/gorilla.py`: 평가 문항 끝에 지시를 붙이고, 모델이 쓴 `계산식:`을 계산해 `정답: 값`을 붙인다.
  식이 없거나 계산할 수 없으면 원래 답을 그대로 둔다(실패는 점수에 그대로 드러난다).
- 도메인을 모른다. 금융이든 다른 분야든 같은 부품을 쓴다.

## 쓰는 법

```bash
python -m genesis.harness.evaluate run --model Qwen/Qwen3-4B --template-kwargs '{"enable_thinking": false}' \
    --data data/applied/fin_v1/eval_test.jsonl --name base-gorilla --pipeline gorilla \
    --batch-size 16 --max-new-tokens 128 --out reports/applied/base-gorilla.json
python -m genesis.harness.evaluate compare reports/applied/base.json reports/applied/base-gorilla.json
```

보고서의 `settings.pipeline`, `pipeline_fallbacks`(식을 못 뽑은 횟수)가 같이 기록된다.

## 아직 모르는 것

- 원본 Qwen3-4B가 지시를 얼마나 따라 `계산식:` 한 줄을 쓰는지(fallback 비율).
- 시스템 프롬프트의 "마지막 줄에 '정답: 값'" 지시와 충돌하는지. 충돌하면 문항 쪽 시스템 문구를 조정한다.
- 다음 단계: 계산 결과를 출처 숫자와 대조하는 검산, 다중 단계 도구 호출. 1단계 결과를 본 뒤에 정한다.
