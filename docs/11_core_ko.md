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

`genesis/serving/endpoint.py`는 표준 라이브러리만 쓰므로 앱이 이 파일 하나만 복사해 써도 된다. 로컬 시험은 Ollama(`http://localhost:11434/v1`)로 같은 API를 쓸 수 있다.

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
