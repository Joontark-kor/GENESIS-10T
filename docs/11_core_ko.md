# GENESIS 코어: 표준은 가중치가 아니라 규약이다

GENESIS는 특정 가중치 하나가 아니라, 어떤 공개 기반 모델 위에서도 같은 방식으로 학습·평가·서빙하게 하는 **규약**이다. 기반 가중치는 바뀔 수 있고, 바뀌어도 그 위의 적용과 앱은 그대로 동작해야 한다.

## 3층 구조

| 층 | 무엇 | 이 저장소에서 |
|---|---|---|
| 1. 기반 가중치 | 공개 모델(실용 트랙), 직접 사전학습한 모델(연구 트랙) | 연구 트랙: `genesis/model.py`, `train.py` 등 |
| 2. GENESIS 코어 | 데이터, 학습기, 평가 하네스, 서빙 규약, 안전 검사 인터페이스 | `genesis/` |
| 3. 적용 | 분야별 LoRA 어댑터와 그 데이터·과제 | `genesis_fin/` (공개 분야). 비공개 분야는 별도 저장소 |

## 코어 모듈

| 모듈 | 역할 |
|---|---|
| `genesis.adapt` | 기반 모델에 LoRA/QLoRA 학습. 응답 토큰만 학습, `run.json`에 기반 모델·리비전·데이터 해시·설정 기록 |
| `genesis.harness` | 모든 모델·어댑터가 같은 형식의 보고서를 낸다 (아래) |
| `genesis.serving` | OpenAI 호환 엔드포인트 클라이언트와 vLLM 멀티 LoRA 실행 명령 |
| `genesis.safety` | 안전 검사의 **인터페이스만**. 판단 규칙은 각 적용이 자기 저장소에 구현한다 |

## 규약 1: 앱은 엔드포인트만 부른다

앱은 GENESIS 모델 코드를 import하지 않는다. 기반 모델 하나에 분야별 어댑터를 이름으로 올리고(vLLM 멀티 LoRA), 앱은 OpenAI 호환 API로 그 이름을 부른다. 그래서 기반을 바꿔도 앱 코드는 바뀌지 않는다.

```bash
python -m genesis.serving.endpoint vllm --base Qwen/Qwen3-4B --adapter fin=runs/applied/fin-v1/adapter
# 출력된 명령으로 서버를 띄운 뒤
python -m genesis.serving.endpoint models --url http://localhost:8000/v1
```

`genesis/serving/endpoint.py`는 표준 라이브러리만 쓰므로 앱이 이 파일 하나만 복사해 써도 된다. 로컬 시험은 Ollama(`http://localhost:11434/v1`)로 같은 API를 쓸 수 있다. vLLM·Ollama가 없는 기계(Colab T4 등)에서는 `python -m genesis.serving.mini_server --model Qwen/Qwen3-4B`가 같은 API를 transformers로 띄운다(시험용, 한 번에 한 요청).

## 규약 2: 평가 보고서 형식이 표준이다

`python -m genesis.harness.evaluate run`은 로컬 모델, 모델+어댑터, 서빙 중인 엔드포인트를 같은 방식으로 평가하고 같은 필드를 남긴다.

| 필드 | 내용 |
|---|---|
| `report_version`, `genesis_version` | 보고서 형식 버전, 만든 GENESIS 버전 |
| `model`, `adapter`, `endpoint`, `data_sha256` | 무엇을 어떤 평가 세트로 쟀나 |
| `accuracy`, `by_kind`, `parsed` | 정확도 전체·유형별, 답을 읽어 낸 비율 |
| `speed`, `cost` | 문항당 시간, 초당 토큰, GPU 시간 × 시간당 가격 |
| `contamination` | `--train-data`를 주면 평가 질문이 학습 데이터에 그대로 있는 비율 |

평가 문항은 `messages`, `gold`와 선택 필드 `kind`, `tolerance`, `scorer`로 이루어진다. 채점기는 코어에 분야 중립적인 `numeric`(허용 오차 안의 숫자)과 `exact`가 있고, 분야 채점기는 `--scorer 모듈:함수`로 붙인다.

### 보고서를 읽는 도구 (2026-11-12)

```bash
# 짝 비교: 같은 문항에서 한쪽만 맞힌 수와 McNemar z (앞 보고서가 낫다 = +, |z|>1.96 ≈ p<0.05)
python -m genesis.harness.evaluate compare reports/a.json reports/b.json
# 여러 seed: 묶음별 평균과 최소-최대
python -m genesis.harness.evaluate seeds base='reports/rep-multi-s*-base.json' pb='reports/rep-multi-s*-pb.json'
# 긴 평가: 문항 묶음마다 OUT.partial.jsonl에 저장, 세션이 끊기면 같은 명령에 --resume을 붙여 이어서
python -m genesis.harness.evaluate run ... --out reports/x.json --resume
```

- 짝 비교는 문항 id로 맞추므로 같은 평가 파일로 만든 보고서끼리만 된다(아니면 "not comparable"). 11-11에 손으로 낸 GSM8K 수치(143 대 85, z≈3.84)를 이 명령으로 다시 낼 수 있다.
- 이어하기는 id와 정답이 같은 문항까지만 믿고, 평가 파일이 바뀌었으면 처음부터 다시 한다. 폴라베어 카운터(재시도·대체 수 등)와 누적 시간·토큰도 저장해 최종 보고서가 한 번에 돌린 것과 같은 필드를 가진다. `--resume` 없이 돌리면 옛 partial 파일은 버린다.

## 규약 3: 버전으로 동결한다

평가 세트를 고정한 채 점수를 기록하면 그 상태를 `vX.Y.Z` 태그로 동결한다. 적용과 앱은 "GENESIS 버전 + 어댑터 + 평가 보고서" 세 가지에 고정해 의존한다. 기반 교체나 코어 변경은 새 버전으로만 들어간다.

## 안전 검사 인터페이스

코어는 판단 규칙을 갖지 않는다. 적용은 `name`, `check_input(messages)`, `check_output(messages, reply)`를 가진 클래스를 자기 저장소에 만들고, 코어는 모델 호출 앞뒤에서 그것을 부른다.

```python
from genesis.safety import load_safety_check, guarded_chat
from genesis.serving import EndpointClient

check = load_safety_check("my_app.safety:Checker")      # 또는 GENESIS_SAFETY_CHECK 환경변수
client = EndpointClient("http://localhost:8000/v1")
result = guarded_chat(check, lambda m: client.chat("my-adapter", m).text, messages)
```

판정(`Verdict`)은 `allow`, `block`, `replace`(출력만), `handoff`(사람에게 넘김) 넷이다. 입력에서 막히면 모델을 아예 부르지 않는다. 검사를 설정하지 않으면 `AllowAll`이 쓰이고, 결과에 "no safety check configured"가 남아 검사가 없다는 사실이 늘 보인다.

## 의존 방향

```
genesis (코어, 공개)  ──▶  genesis_fin (공개 적용)
                      ──▶  비공개 적용 저장소들
```

코어는 어떤 적용도 import하지 않는다. `tests/test_boundaries.py`가 이것을 검사한다. 적용 쪽 요구는 적용 저장소에서 처리하고, 코어에는 분야 중립적인 기능으로만 들어온다.


## 계산이 필요한 질문: `CalcClient` (2026-11-08)

앱이 숫자 질문을 할 때는 점수가 아니라 **어떤 식으로 나온 값인지**가 필요하다. `genesis.serving.calc.CalcClient`는 엔드포인트에 질문하고,
모델이 쓴 식을 계산기(`genesis.tools.calculate`)로 계산해 돌려준다. 결과(`CalcResult`)에는 값(`answer`), 식(`formula`), 계산기 사용 여부,
재시도 여부, 그리고 확인할 점(`warnings`: 식 없이 답함 / 질문에 없는 숫자 / 증감률 방향 의심)이 들어 있다.

```python
from genesis.serving import EndpointClient
from genesis.serving.calc import CalcClient
result = CalcClient(EndpointClient("http://localhost:8000/v1"), "fin").ask([{"role": "user", "content": "정가 12,000원, 15% 할인가는?"}])
print(result.answer, result.formula, result.warnings)
```

- **언제 부르나(v0.3.0 범위):** 값이 문제 안에 주어진 정형 계산 질문(금융 비율, 일상 수치, 다단계 정형 계산)에서만 부른다. 질문 종류를 모르는 자리에서는 기본 모델을 쓴다. 서술형 수학 문장제(한국어 GSM8K 전체 64.1% → 59.7%)와 계산 없는 질문(85.1% → 79.2%)에서는 기준선보다 낮다(근거: `docs/12_polarbear_ko.md`).
- 앱은 `warnings`가 비어 있을 때만 값을 그대로 쓰고, 아니면 사람에게 확인을 요청하거나 식을 보여주는 식으로 쓸 수 있다.
- 실험 옵션(제외 검사 등)은 평가 하네스 전용이다. 서빙 쪽에는 검증된 기본 동작과 `--distractor-hint`만 둔다.
- 비공개 컴패니언 쪽 연결(대화 설계, 안전 파이프라인)은 이 저장소에 두지 않는다. 이 클라이언트는 엔드포인트만 부른다.
