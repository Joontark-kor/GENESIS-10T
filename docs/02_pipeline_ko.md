# 실제 한국어 데이터로 처음부터 끝까지

이 문서는 출처가 기록된 원천 데이터에서 시작해 토크나이저, 토큰 샤드, 학습, 평가까지 이어지는 전체 흐름을 설명한다. 모든 단계는 앞 단계의 해시를 기록하므로 어떤 체크포인트가 어떤 데이터와 토크나이저로 만들어졌는지 추적할 수 있다.

```
sources.json ──A──▶ data/clean/v1 ──B──▶ tokenizers/ko32k-v1.json
  (권리 기록)         (정제 텍스트,            │
                      DATA_CARD.md)          │
                           └────────C─────────┴──▶ data/tokens/ko32k-v1 ──D──▶ checkpoints/… ──E──▶ 평가·오염 보고
```

## 0. 원천 데이터 받기 (직접 해야 함)

이 저장소는 데이터를 대신 내려받지 않는다. 각 소스의 이용 조건을 직접 읽고 받는다.

| 소스 | 받는 방법 | 라이선스 / 권리 근거 | 주의 |
|---|---|---|---|
| 한국어 위키백과 | `dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2` 다운로드 후 `pip install wikiextractor` → `python -m wikiextractor.WikiExtractor --json -o data/raw/kowiki kowiki-latest-pages-articles.xml.bz2` | CC BY-SA 4.0 / `open-license` | 재배포 시 동일조건 유지, 출처 표기 |
| 법령·판결문 | 국가법령정보 공동활용 Open API(open.law.go.kr, 인증키 신청 필요)로 받아 법령 1건당 `.txt` 1개로 저장 → `data/raw/law/` | 저작권법 제7조 / `statutory-unprotected` | 출판사 해설·주석은 보호 대상일 수 있으니 공식 원문만 |
| FineWeb2 한국어 | `pip install -U "huggingface_hub[cli]" pyarrow` → `huggingface-cli download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/kor_Hang/train/*" --local-dir data/raw/fineweb2` | ODC-By 1.0 / `open-license` | ODC-By는 편집물 라이선스. 개별 웹페이지 저작권은 별개이므로 `commercial_use: unknown`으로 기록 |
| 공공누리 1유형 자료 | 각 기관 사이트 | KOGL Type 1 / `public-sector-license` | 2~4유형(상업 금지·변경 금지)은 제외 |
| 모두의 말뭉치, AI Hub | 이용 신청 | 약관별 상이 / `contract` | 목적·재배포·해외 반출 제한을 개별 확인. 모두의 말뭉치 일부 전자책은 2026년 저작권 판결 대상이었음 |

처음에는 **위키백과 + 법령**만으로 시작하는 것을 권한다. 권리 관계가 가장 명확하고, 100M 모델 실험에 필요한 수억~수십억 토큰 규모를 만들 수 있다. 정확한 토큰 수는 2단계 실행 결과로 확인한다.

## 1. 소스 등록부 작성

```bash
cp configs/sources.example.json configs/sources.json
```

받은 소스만 남기고 경로를 맞춘다. 필수 필드는 `name`, `type`(txt/jsonl/parquet), `path`, `license`, `rights_basis`다. `license`가 비어 있거나 `unknown`이면 파이프라인이 실행을 거부한다. 소스별 필터는 `filters`로 덮어쓸 수 있다(예: 짧은 문장이 많은 소스는 `{"min_chars": 20}`). 처음 실험할 때는 `max_docs`로 문서 수를 제한한다.

## 2. 정제 (A단계)

```bash
python -m genesis.data.clean --sources configs/sources.json --output data/clean/v1
```

순서: NFC 정규화 → 품질 필터 → 정확 중복 제거 → MinHash 유사 중복 제거 → 개인정보 마스킹 → 내용 해시로 train/val/test 분할(기본 98/1/1).

**합격 기준(직접 확인):**
- `data/clean/v1/DATA_CARD.md`의 소스별 제거 사유 비율이 상식적인지
- `clean_manifest.json`의 `removed_examples`에서 잘못 지워진 좋은 문서가 없는지
- `zcat data/clean/v1/train/part-00000.jsonl.gz | shuf -n 20`으로 무작위 문서 20개를 직접 읽어보기

필터가 너무 공격적이면 `QualityConfig` 기본값이나 소스별 `filters`를 조정하고 다시 실행한다. 버전 디렉터리(`v1`, `v2`)를 바꿔가며 보관한다.

> MinHash 색인은 메모리에 올라간다(문서당 약 0.5KB). 수천만 문서 규모에서는 소스별로 나눠 돌리거나 datatrove 같은 분산 도구로 옮길 시점이다.

## 3. 토크나이저 학습 (B단계)

```bash
python -m genesis.train_tokenizer --input data/clean/v1 --vocab-size 32768 \
    --max-docs 2000000 --output tokenizers/ko32k-v1.json
```

train 분할로만 학습하고, val 분할에서 바이트 토크나이저와 비교한 표를 출력한다. 같은 표가 `tokenizers/ko32k-v1.json.meta.json`에 저장된다. 다른 어휘 크기도 비교해 본다.

```bash
python -m genesis.tokenizer_report --input data/clean/v1 --split val \
    --tokenizers byte tokenizers/ko32k-v1.json tokenizers/ko64k-v1.json
```

**합격 기준:** 왕복 복원 100%, 한글 음절당 토큰 수가 바이트(약 3.0~3.4) 대비 크게 낮을 것. NSMC(영화 리뷰 20만 건, CC0)로 시험한 8k 어휘에서 0.67이 나왔다. 위키·법령 같은 긴 문서에서는 더 낮아지는 것이 정상이다. 토크나이저 JSON과 meta 파일은 저장소에 커밋해 재현성을 확보한다.

## 4. 토큰 샤드 (C단계)

```bash
python -m genesis.data.tokenize --input data/clean/v1 \
    --tokenizer tokenizers/ko32k-v1.json --output data/tokens/ko32k-v1
```

`manifest.json`에 토크나이저 해시가 기록되고, 학습·평가 시 이 해시가 다르면 실행이 중단된다. 어휘가 65,536개 이하면 uint16, 그 이상이면 uint32로 자동 저장된다.

## 5. 학습 (D단계)

먼저 규모와 비용을 계산한다.

```bash
python -m genesis.estimate --config configs/dense-100m.json --tokens 2e9 --gpu-tflops 989 --mfu 0.35
```

단일 GPU:

```bash
python -m genesis.train --config configs/dense-100m.json --data data/tokens/ko32k-v1 \
    --out-dir checkpoints/dense-100m --batch-size 16 --grad-accum 4 --steps 15000 \
    --lr 6e-4 --eval-every 500 --save-every 1000 --peak-tflops 989
```

한 노드 여러 GPU(DDP). 전역 배치 = GPU 수 × `--batch-size` × `--grad-accum`이므로, 위 단일 GPU 예시와 같은 64개 시퀀스(약 13만 토큰)를 쓰려면 8 GPU에서 `--batch-size 8 --grad-accum 1`로 맞춘다:

```bash
torchrun --standalone --nproc_per_node 8 -m genesis.train --config configs/dense-100m.json \
    --data data/tokens/ko32k-v1 --out-dir checkpoints/dense-100m \
    --batch-size 8 --grad-accum 1 --steps 15000 --lr 6e-4 --save-every 1000 --peak-tflops 989
```

중단되면 같은 명령에 `--resume`을 붙인다. 같은 스텝에서 같은 배치를 다시 뽑도록 설계되어 있어 손실 곡선이 끊김 없이 이어진다(CPU에서 자동 테스트로 확인됨). 시간 제한이 있는 작업에서는 `--stop-after N`으로 끊고 이어서 돌린다.

`checkpoints/<run>/metrics.jsonl`에 스텝별 손실, 학습률, grad norm, tokens/s, MFU, val loss와 **val bpb**가 쌓인다. 토크나이저가 다른 모델끼리는 bpb로만 비교한다.

## 6. 평가와 오염 검사 (E단계)

```bash
python -m genesis.evaluate --checkpoint checkpoints/dense-100m/model.pt \
    --data data/tokens/ko32k-v1 --split val
```

벤치마크 점수를 공개하기 전에 해당 벤치마크 문항이 학습 데이터에 섞였는지 확인한다. 벤치마크를 JSONL(한 줄에 한 문항)로 받아 문항 텍스트가 들어 있는 필드를 지정한다.

```bash
python -m genesis.contamination --train data/clean/v1 \
    --benchmark data/bench/kobest_boolq.jsonl --fields question paragraph \
    --report reports/contamination_kobest_boolq.json --blocklist reports/blocklist.txt
python -m genesis.data.clean --sources configs/sources.json --output data/clean/v2 \
    --exclude-sha256 reports/blocklist.txt
```

KoBEST·HAE-RAE·CLIcK 같은 다운스트림 점수는 HRET 또는 lm-evaluation-harness로 측정한다. 이를 위한 Hugging Face 형식 내보내기는 다음 작업 목록에 있다.
