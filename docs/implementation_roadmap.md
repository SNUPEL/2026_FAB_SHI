# implementation roadmap

이 문서는 연구계획서의 개발 흐름을 현재 스켈레톤 코드 관점에서 다시 정리한 것입니다.

## 1단계. 문제 모델링

현재 완료:
- 병렬 이기종 설비 문제 구조 반영
- 작업 / 설비 / 후공정 베이 자료형 정의
- action = 작업-설비 pair 스켈레톤 정의

코드:
- `Environment/data.py`
- `Environment/environment.py`
- `Environment/simulation.py`

## 2단계. 제약조건 정의

현재 완료:
- 하드 / 소프트 제약 분리
- config on/off 구조
- 제약 함수 등록 구조

코드:
- `Environment/constraints/base.py`
- `Environment/constraints/registry.py`
- `Environment/constraints/machine_rules.py`
- `Environment/constraints/downstream_rules.py`
- `Environment/constraints/soft_rules.py`
- `config.yaml`

## 3단계. 택트타임 분석

현재 완료:
- 분석기 스켈레톤만 구현

향후 해야 할 일:
- 실적 데이터 컬럼 정의
- 설비 타입별 처리시간 모델
- 보정계수 산출

코드:
- `Utils/tact_time.py`

## 4단계. 학습 데이터 생성

현재 완료:
- 시나리오 생성기 스켈레톤 구현

향후 해야 할 일:
- 작업 pool 크기 생성
- 긴급도 / 납기 / 베이 조합 생성
- 설비 상태 랜덤화

코드:
- `Utils/scenario_generator.py`

## 5단계. 학습 알고리즘

현재 완료:
- REINFORCE / PPO / Self-labeling 자리 구현

향후 해야 할 일:
- 실제 network 연결
- rollout buffer
- loss 계산
- optimizer

코드:
- `Train/algorithms.py`
- `Train/runner.py`

## 6단계. 비교/검증

현재 완료:
- 휴리스틱 3종 구현
- simulate / trace / eval 실행 경로 구현

코드:
- `Agent/heuristics.py`
- `main.py`

## 7단계. 앞으로 바로 해야 할 일

우선순위는 아래 순서가 맞습니다.

1. 실제 데이터 컬럼 정의
2. 택트타임 모델 구체화
3. 하드/소프트 제약 현업 기준 확정
4. reward 계수 설계
5. PPO 또는 self-labeling 실제 구현
