# 지시학습(SFT): 사전학습 모델을 대화형으로

사전학습한 dense 모델 위에 질문-답변 데이터를 학습해 대화형 모델을 만든다. 손실은 **assistant 응답 토큰과 응답 끝 표시(`<|im_end|>`)에만** 걸리므로, 모델은 사용자 말투를 흉내 내지 않고 답한 뒤 멈추는 법을 배운다.

## 대화 형식

토크나이저에 예약해 둔 토큰을 쓰는 ChatML 형식이다. Hugging Face로 내보내면 같은 형식이 `chat_template`으로 들어가고, 토큰 단위로 일치하는 것을 테스트로 확인했다.

```
<|im_start|>system
친절하게 답하세요.<|im_end|>
<|im_start|>user
한국의 수도는?<|im_end|>
<|im_start|>assistant
서울입니다.<|im_end|>
```

## 1. 데이터 준비

JSONL 한 줄에 대화 하나. 두 형식을 받는다.

```json
{"messages": [{"role": "user", "content": "한국의 수도는?"}, {"role": "assistant", "content": "서울입니다."}]}
{"instruction": "다음 문장을 존댓말로 바꿔줘", "input": "밥 먹었어?", "output": "식사하셨어요?"}
```

```bash
python -m genesis.data.sft --input "data/raw/sft/*.jsonl" --tokenizer tokenizers/ko32k-v1.json \
    --output data/sft/v1 --source <데이터 이름> --license <라이선스> --rights-basis open-license \
    --commercial-use unknown --max-length 2049
```

`--max-length`는 모델의 `context_length + 1`로 맞춘다. 더 긴 대화는 버리고 개수를 기록한다. 개인정보 마스킹, 중복 제거, train/val 분할이 자동으로 된다.

**데이터 선택 기준:** 라이선스가 명시된 공개 데이터, 직접 작성한 데이터, 또는 출력물 학습 사용을 허용하는 모델로 만든 합성 데이터. 처음에는 수천~수만 건이면 충분하다.

## 2. 학습

```bash
python -m genesis.train --config configs/dense-100m.json --data data/sft/v1 \
    --init-from checkpoints/dense-100m/model.pt --out-dir checkpoints/dense-100m-sft \
    --steps 2000 --batch-size 16 --lr 5e-5 --warmup-steps 50 --eval-every 200 --save-every 500
```

- 학습률은 사전학습의 1/10 정도에서 시작한다. 너무 높으면 사전학습 지식을 잃는다.
- `--init-from` 체크포인트의 설정과 토크나이저가 다르면 실행이 거부된다.
- `val_bpb`는 val 대화의 응답 부분에 대해서만 계산된다.

## 3. 대화해 보기

```bash
python -m genesis.generate --checkpoint checkpoints/dense-100m-sft/model.pt --chat \
    --prompt "조선의 첫 번째 왕은 누구야?" --tokens 120
```

## 한계

100M 모델은 SFT 후에도 사실 정확도가 낮다. 이 단계의 목적은 형식(질문에 답하고 멈추기)과 파이프라인 검증이다. 안전성 평가와 선호 학습(DPO 등)은 아직 구현되어 있지 않다.
