# 다음 단계와 역할 분담 (2026-10-29)

## 이번 작업으로 된 것

| 로드맵 | 상태 | 내용 |
|---|---|---|
| Gate 1 데이터 | 코드 완료 | 권리 기록 필수 등록부, NFC 정규화, 품질 필터(제거 사유 기록), 개인정보 마스킹, 정확·유사 중복 제거, 문서 단위 분할, 데이터 카드 |
| Gate 1 토크나이저 | 코드 완료 | 한국어 우선 byte-level BPE, 음절당 토큰 수 비교 리포트, 해시로 데이터·체크포인트와 연결 |
| Gate 2 학습기 | 코드 완료 | RoPE·RMSNorm·GQA·SDPA, dense/MoE 선택, LR 스케줄, grad accumulation, BF16, DDP, 정확한 재개, bpb·MFU 기록 |
| Gate 2 평가 | 일부 | val/test loss·bpb, 벤치마크 오염 검사와 차단 목록. 다운스트림 벤치마크 연동은 아직 |

자동 테스트 47개가 CPU에서 통과한다. GPU에서는 아직 실행하지 않았다.

### 소규모 시험 결과 (참고용)

NSMC 영화 리뷰(CC0, 약 18만 문서 정제 후)로 같은 구조의 약 1~2M 파라미터 모델을 같은 스텝 수(400 스텝 × 16 시퀀스 × 128 토큰)만큼 학습했다.

| 토크나이저 | 한글 음절당 토큰 | 바이트당 토큰 | val bits-per-byte |
|---|---:|---:|---:|
| 바이트 | 3.42 | 1.00 | 2.41 |
| BPE 8k (NSMC train으로 학습) | 0.67 | 0.19 | 2.06 |

같은 문맥 길이에 약 5배 많은 한국어가 들어가고, bpb도 약 15% 낮다. 단, BPE 모델은 임베딩 때문에 파라미터가 더 많고 학습 시간도 약 2배였으므로 엄밀한 동일 연산 비교는 아니다. 실제 판단은 위키·법령 코퍼스와 100M 모델에서 다시 측정해야 한다.

## 직접 하셔야 할 일 (순서대로)

1. **이 브랜치를 GitHub에 반영**
   이 세션은 저장소에 push 권한이 없어 git bundle로 전달했다. 저장소 폴더에서:
   ```bash
   git fetch /경로/genesis-gate1-2.bundle gate1-2-korean-pipeline:gate1-2-korean-pipeline
   git checkout gate1-2-korean-pipeline
   pip install -e ".[dev]" && python -m pytest -q
   git push -u origin gate1-2-korean-pipeline   # 그다음 GitHub에서 PR → main 병합
   ```
   claude.ai 설정의 GitHub 연결을 켜 두면 다음부터는 브랜치와 PR을 직접 만들 수 있다.

2. **목표 결정**: (a) 처음부터 만드는 역량·공개 연구, (b) 특정 용도의 한국어 모델(제품). (b)라면 사전학습은 100M~1B 검증에서 멈추고 공개 모델 위에 도메인 지속학습·파인튜닝을 하는 것이 빠르다. 아래 3~7은 두 경우 모두 유효하다.

3. **데이터 확보**: 위키백과와 법령부터 받아 `configs/sources.json`을 작성하고 정제한다([02_pipeline_ko.md](02_pipeline_ko.md) 0~2절). 자동화할 수 없는 부분은 **데이터 카드와 무작위 샘플을 사람이 직접 읽는 일**이다.

4. **권리 검토**: 소스 목록이 정해지면 상업 공개 전에 법률 검토를 받는다. 특히 웹 크롤 기반(FineWeb2)과 계약 기반(모두의 말뭉치, AI Hub) 소스.

5. **GPU 확보**: CPU로는 `ko-tiny-cpu`까지만 의미가 있다. 아래는 `genesis.estimate` 기준 H100 BF16 989 TFLOPS, MFU 40% 가정치다.

   | 실험 | 파라미터 | 학습 토큰 | 대략 H100 GPU-시간 |
   |---|---:|---:|---:|
   | dense-100m | 101M | 2B | 1~2 |
   | moe-8x-100m-active | 271M (활성 101M) | 2B | 1~2 (+라우팅 오버헤드) |
   | dense-300m | 304M | 6~7B | 약 10 |
   | dense-1b | 969M | 30B | 약 130 (8장이면 약 16시간) |

   실제 MFU는 측정해야 한다(`--peak-tflops`를 주면 로그에 찍힌다). 클라우드 요금은 업체마다 달라 견적을 받아 비교한다.

6. **첫 GPU 실험**: `dense-100m`을 2B 토큰으로 학습하고 `metrics.jsonl`과 `DATA_CARD.md`를 공유해 주면 다음 판단(학습률, 데이터 혼합, 규모)을 같이 할 수 있다.

7. **벤치마크 데이터 확보**: KoBEST, HAE-RAE, CLIcK를 받아 두고, 학습 전에 `genesis.contamination`으로 오염을 검사한다.

## 이어서 개발할 것 (요청하면 바로 진행)

1. Hugging Face(Llama 호환) 형식 내보내기 → HRET / lm-evaluation-harness로 KoBEST·HAE-RAE·CLIcK 측정
2. 스케일링 사다리 실행 스크립트(30M/100M/300M)와 결과 그래프, dense vs MoE 비교 리포트
3. 생성용 KV cache
4. FSDP2(1B 이상, GPU 메모리를 넘는 모델)와 MoE 전문가 병렬화
5. 지시학습(SFT): 예약 토큰 `<|im_start|>`, `<|im_end|>`를 쓰는 대화 형식과 손실 마스킹
6. 수천만 문서 이상에서 정제 단계를 datatrove 등 분산 도구로 이전
