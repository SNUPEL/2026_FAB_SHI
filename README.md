# pmsp

절단 공정용 `PMSP(Parallel Machine Scheduling Problem)` 기반 스케줄링 프로젝트입니다.

이 저장소는 연구계획서의 큰 흐름을 바로 코드로 옮길 수 있도록 만든 **실행 가능한 스켈레톤 + 확장 가능한 프레임워크**입니다.  
즉, 단순 문서용 뼈대가 아니라 아래가 실제로 동작합니다.

- 병렬 이기종 설비 환경 `reset / step`
- 휴리스틱 실행
- PPO / REINFORCE / Self-labeling 학습 루프
- category / hard / soft / override 제약 구조
- calendar 제약
- 설비별 운영시간 / 계획 정지 / 고장
- 샘플 시나리오 기반 시뮬레이션 / trace / eval

다만 아직 **현업 실데이터 기반 최종 모델**은 아닙니다.  
실제 택트타임 산출식, 상세 셋업 룰, 정반 분할 물리 로직, 후공정 상세 반출 로직 등은 이후 데이터가 들어오면 채우는 구조입니다.

---

## 1. 프로젝트 목적

이 프로젝트는 절단 공정을 아래 문제로 정의합니다.

- 여러 절단 설비가 동시에 존재
- 설비마다 가능한 계열 / 두께 / 길이 / 속도가 다름
- 작업마다 후공정 베이와 우선순위가 존재
- 설비 부하 평준화도 중요
- 운영일 / 점심시간 / 설비 고장 같은 캘린더 제약도 중요

즉, 본질적으로는 **비관련 병렬기계 스케줄링 + 설비 가능 제약 + 후공정 연계 제약** 문제입니다.

---

## 2. 현재 구현 수준

현재 상태를 한 문장으로 정리하면 아래와 같습니다.

> 학습과 휴리스틱을 붙일 수 있는 PMSP형 절단 스케줄링 환경은 구현 완료  
> 단, 현업 데이터 기반 상세 택트타임 / 상세 물리 제약은 추후 채움

상세 상태표는 아래 문서를 보면 됩니다.

- [docs/implementation_status.md](docs/implementation_status.md)

---

## 3. 폴더 구조

```text
pmsp/
├─ main.py
├─ config.yaml
├─ Agent/
│  ├─ heuristics.py
│  └─ policy.py
├─ Environment/
│  ├─ data.py
│  ├─ environment.py
│  ├─ reward.py
│  ├─ simulation.py
│  ├─ state.py
│  └─ constraints/
│     ├─ base.py
│     ├─ calendar_rules.py
│     ├─ downstream_rules.py
│     ├─ machine_rules.py
│     ├─ registry.py
│     └─ soft_rules.py
├─ Train/
│  ├─ runner.py
│  ├─ algorithm/
│  │  ├─ common.py
│  │  ├─ factory.py
│  │  ├─ ppo.py
│  │  ├─ reinforce.py
│  │  └─ self_labeling.py
│  └─ network/
│     ├─ feature_builder.py
│     ├─ gnn.py
│     ├─ mlp.py
│     └─ policy_value.py
├─ Utils/
│  ├─ config.py
│  ├─ io.py
│  ├─ scenario_generator.py
│  └─ tact_time.py
├─ input/
│  └─ sample_scenario.yaml
├─ output/
└─ docs/
```

---

## 4. 핵심 개념

### 4.1 action

현재 기본 action은 `job_machine_pair`입니다.

즉, 한 step에서 고르는 것은:

- 작업 1개
- 설비 1개

의 쌍입니다.

예:
- `job_003 @ laser_01`
- `job_008 @ plasma_02`

---

### 4.2 state

환경이 매 step마다 보는 상태는 크게 두 부분입니다.

- 남은 작업 상태
- 설비 / 후공정 / 캘린더 상태

포함 예:
- 남은 작업 수
- 설비별 현재 부하
- 설비별 현재 가용 시각
- 후공정 bay 적치량
- 현재 날짜 / 현재 시각
- 현재 가능한 action 목록

---

### 4.3 reward

현재 reward는 아래를 조합합니다.

- makespan 증가 벌점
- 설비 간 부하 불균형 벌점
- 우선 작업 bonus
- soft 제약 penalty

즉, “빨리 끝내되, 너무 한 설비에 몰리지 말고, 소프트 제약도 가능하면 지키는 방향”입니다.

---

## 5. 제약 구조

이 프로젝트의 제약은 4개 축으로 관리합니다.

1. `category`
2. `hard`
3. `soft`
4. `override`

### 5.1 category

사람이 이해하기 쉬운 큰 분류입니다.

- `machine`
- `capacity`
- `downstream`
- `priority`
- `preference`
- `layout`
- `calendar`

### 5.2 hard

위반하면 후보 action에서 제거됩니다.

예:
- 두께 범위 위반
- 정반 길이 초과
- 설비 비활성
- 적치량 초과

### 5.3 soft

위반해도 후보는 남고 penalty만 부여됩니다.

예:
- 부하 평준화 선호
- 납기 긴급도 반영
- 선호 설비 타입

### 5.4 override

날짜별 운영 예외를 반영합니다.

예:
- 특정 날짜 설비 용량 50%
- 특정 날짜 설비 비가동
- 특정 날짜 하루 작업 수 제한

상세 설명:
- [docs/constraint_catalog.md](docs/constraint_catalog.md)
- [docs/config_guide.md](docs/config_guide.md)

---

## 6. calendar 기능

현재 calendar 계열에서 지원하는 기능은 아래와 같습니다.

- 휴무일
- 반일 가동
- 점심시간
- 공장 전체 shutdown window
- 설비별 운영시간
- 설비별 계획 정지
- 설비별 고장

중요:
- 현재는 **작업 시작 시점 기준**입니다.
- 작업이 이미 시작된 뒤 점심시간/고장이 끼어드는 `preemption / resume`은 아직 없습니다.

상세 설명:
- [docs/calendar_constraints.md](docs/calendar_constraints.md)

---

## 7. 제약을 추가하는 방법

초보자 기준으로는 아래 3단계만 기억하면 됩니다.

1. 제약 함수 1개 작성
2. `registry.py`에 등록
3. `config.yaml`에서 켜기

즉, simulation 본문을 매번 수정하지 않아도 됩니다.

상세 가이드:
- [docs/constraint_add_guide.md](docs/constraint_add_guide.md)

---

## 8. 학습 알고리즘

현재 선택 가능한 알고리즘:

- `ppo`
- `reinforce`
- `self_labeling`

선택 위치:
- [config.yaml](config.yaml)

예:

```yaml
train:
  episodes: 10
  algorithm: ppo
```

네트워크는 `GNN + MLP` 기반 정책/가치 구조를 사용합니다.

관련 코드:
- [Train/algorithm/ppo.py](Train/algorithm/ppo.py)
- [Train/algorithm/reinforce.py](Train/algorithm/reinforce.py)
- [Train/algorithm/self_labeling.py](Train/algorithm/self_labeling.py)
- [Train/network/policy_value.py](Train/network/policy_value.py)

참고 논문/코드:
- [docs/rl_references.md](docs/rl_references.md)

---

## 9. 빠른 실행

### 9.1 현재 설정 요약

```bash
python3 main.py show-config --config config.yaml
```

### 9.2 휴리스틱 시뮬레이션

```bash
python3 main.py simulate --config config.yaml
```

### 9.3 step-by-step trace

```bash
python3 main.py trace --config config.yaml
```

### 9.4 택트타임 분석 스켈레톤

```bash
python3 main.py analyze-tact --config config.yaml
```

### 9.5 시나리오 생성

```bash
python3 main.py generate-scenario --config config.yaml --duplicate-jobs 2 --output-path output/generated_scenario.yaml
```

### 9.6 학습

```bash
python3 main.py train --config config.yaml
```

### 9.7 평가

```bash
python3 main.py eval --config config.yaml
```

---

## 10. 샘플 설정 예시

예를 들어 `calendar`와 `machine` 제약을 켜고 싶다면:

```yaml
constraints:
  categories:
    machine: true
    calendar: true

  hard_enabled:
    calendar_open: true
    machine_calendar_open: true
    machine_breakdown: true
    machine_enabled: true
    family_eligibility: true
    thickness_range: true
    table_length_limit: true
```

설비별 운영시간 예시:

```yaml
calendar:
  enable_machine_operating_windows: true
  machine_operating_windows:
    default:
      laser_01:
        - "08:00-18:00"
      plasma_01:
        - "00:00-24:00"
```

---

## 11. 아직 비어 있는 것

아직 구현은 되어 있지만 실제 로직이 비어 있거나 단순화된 부분:

- 실데이터 기반 택트타임 산출식
- 셋업 상세 규칙
- 정반 3분할 / 길이 분할 물리 제약
- 후공정 반입/반출 시간 흐름
- preemption / resume
- 2D 레이아웃 상세 모델

즉, 지금은 **연구계획서 전체를 따라갈 수 있는 구조는 완료**되었고,  
**현업 상세값과 물리 세부는 앞으로 채우는 단계**입니다.

---

## 12. 문서 안내

- 문제 정의: [docs/problem_definition.md](docs/problem_definition.md)
- 코드 매핑: [docs/research_plan_to_code_map.md](docs/research_plan_to_code_map.md)
- 구현 상태: [docs/implementation_status.md](docs/implementation_status.md)
- 구현 로드맵: [docs/implementation_roadmap.md](docs/implementation_roadmap.md)
- action 설명: [docs/action_space_notes.md](docs/action_space_notes.md)
- 제약 설명: [docs/constraint_catalog.md](docs/constraint_catalog.md)
- 제약 추가 가이드: [docs/constraint_add_guide.md](docs/constraint_add_guide.md)
- calendar 설명: [docs/calendar_constraints.md](docs/calendar_constraints.md)
- config 설명: [docs/config_guide.md](docs/config_guide.md)

---

## 13. 한 줄 요약

이 저장소는 **절단 공정 PMSP형 스케줄링 문제를 위한 실행 가능한 연구용 프레임워크**이며,  
**초보자도 제약과 운영 규칙을 함수/설정 단위로 추가할 수 있게 설계된 코드베이스**입니다.
