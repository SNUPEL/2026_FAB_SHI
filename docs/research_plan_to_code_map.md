# research plan to code map

연구계획서의 주요 항목을 현재 코드 구조에 대응시킨 문서입니다.

## 1. 절단 장비별 택트타임 분석 및 산출 로직

현재 상태:
- 아직 placeholder 단계
- 실적 데이터가 없으므로 기본 인터페이스만 구현

향후 연결 위치:
- `Utils/tact_time.py`
- `Job.estimate_total_minutes()` 또는 별도 estimator

## 2. 산출 로직 기반 학습 데이터 생성 환경

현재 상태:
- 샘플 시나리오 yaml 존재
- 생성기 스켈레톤 구현

향후 연결 위치:
- `Utils/scenario_generator.py`
- `input/` 아래 생성된 episode 시나리오 저장

## 3. 설비 배정 및 작업 순서 결정

현재 상태:
- 구현됨
- 환경은 `작업-설비 pair 선택` 형태
- reset/step 인터페이스 구현
- 가능한 액션이 없으면 다음 완료 시점으로 자동 전진

코드 위치:
- `Environment/simulation.py`
- `Environment/environment.py`

## 4. 제약/우선순위 반영 DRL 스켈레톤

현재 상태:
- 하드 / 소프트 제약 구조 구현
- reward 기본 구현
- train/eval runner 기본 구현
- config 기반 on/off 구현

코드 위치:
- `Environment/constraints/`
- `Environment/reward.py`
- `Train/runner.py`
- `config.yaml`

## 5. 휴리스틱 비교/검증

현재 상태:
- SPT
- priority
- load_balance

코드 위치:
- `Agent/heuristics.py`

## 6. 실행 진입점

현재 상태:
- 구현됨

코드 위치:
- `main.py`

지원 명령:
- `show-config`
- `simulate`
- `trace`
- `analyze-tact`
- `generate-scenario`
- `train`
- `eval`
