# constraint add guide

이 문서는 **초보자 기준으로 새 제약조건 1개를 추가하는 방법**을 설명합니다.

이 프로젝트의 제약 시스템은 일부러 단순하게 만들었습니다.

핵심은 아래 3단계입니다.

1. 제약 함수 1개 작성
2. `registry.py`에 등록
3. `config.yaml`에서 켜기

즉, 대부분의 경우 **시뮬레이션 본문은 수정하지 않아도 됩니다.**

---

## 1. 먼저 알아야 할 파일

제약 관련 핵심 파일은 아래입니다.

- 공통 자료형: [Environment/constraints/base.py](../Environment/constraints/base.py)
- 제약 등록부: [Environment/constraints/registry.py](../Environment/constraints/registry.py)
- 설비/용량 제약: [Environment/constraints/machine_rules.py](../Environment/constraints/machine_rules.py)
- 후공정 제약: [Environment/constraints/downstream_rules.py](../Environment/constraints/downstream_rules.py)
- 소프트 제약: [Environment/constraints/soft_rules.py](../Environment/constraints/soft_rules.py)
- 캘린더 제약: [Environment/constraints/calendar_rules.py](../Environment/constraints/calendar_rules.py)
- 설정 파일: [config.yaml](../config.yaml)

---

## 2. 제약 함수의 기본 형식

모든 제약 함수는 아래 형식을 따릅니다.

```python
def check_example_rule(context: ConstraintContext) -> ConstraintResult:
    passed = True
    reason = ""
    penalty = 0.0
    return ConstraintResult("example_rule", passed, reason, penalty=penalty)
```

여기서 의미는 다음과 같습니다.

- `context`
  - 현재 배정하려는 작업/설비/상태 정보가 들어 있습니다.
- `passed`
  - `True`: 제약 만족
  - `False`: 제약 위반
- `reason`
  - 왜 위반했는지 설명
- `penalty`
  - 소프트 제약일 때만 주로 사용

---

## 3. 하드 제약과 소프트 제약의 차이

### 3.1 하드 제약

- 위반하면 후보에서 제거됩니다.
- 즉, 아예 선택할 수 없습니다.

예:
- 두께 범위 위반
- 정반 길이 초과
- 적치량 초과
- 설비 비가동

---

### 3.2 소프트 제약

- 위반해도 후보는 남아 있습니다.
- 대신 penalty를 받아 점수가 불리해집니다.

예:
- 가능한 레이저를 우선 쓰고 싶음
- 이미 바쁜 설비는 피하고 싶음
- 후공정 우선순위를 가능하면 맞추고 싶음

---

## 4. 제약 추가 절차

아래 순서대로 하면 됩니다.

### Step 1. 어디에 넣을지 결정

제약 종류에 따라 파일을 고릅니다.

- 설비/용량 관련: `machine_rules.py`
- 후공정/적치 관련: `downstream_rules.py`
- 점수/선호 관련: `soft_rules.py`
- 운영일/고장 관련: `calendar_rules.py`

규칙이 애매하면, 일단 가장 가까운 파일에 넣고 나중에 정리해도 됩니다.

---

### Step 2. 함수 작성

예시로 하드 제약 1개를 추가해보겠습니다.

조건:
- 두께가 40 초과인데
- 설비 타입이 `laser`이면 불가

```python
from .base import ConstraintContext, ConstraintResult


def check_heavy_plate_block(context: ConstraintContext) -> ConstraintResult:
    """두꺼운 재질은 레이저에 배정하지 못하도록 하는 예시 규칙."""

    if context.job.thickness > 40 and context.machine.machine_type == "laser":
        return ConstraintResult(
            "heavy_plate_block",
            False,
            "laser cannot process thickness > 40",
        )

    return ConstraintResult("heavy_plate_block", True, "")
```

이 함수는 매우 단순합니다.

- 조건을 만족하면 `False`
- 아니면 `True`

---

### Step 3. registry.py에 등록

[Environment/constraints/registry.py](../Environment/constraints/registry.py)에 2군데 추가합니다.

#### 3.1 RULES에 등록

```python
RULES = {
    ...
    "heavy_plate_block": check_heavy_plate_block,
}
```

#### 3.2 RULE_CATEGORIES에 등록

```python
RULE_CATEGORIES = {
    ...
    "heavy_plate_block": "machine",
}
```

여기서 category는 사람이 이해하기 쉬운 큰 분류입니다.

자주 쓰는 category:

- `machine`
- `capacity`
- `downstream`
- `priority`
- `preference`
- `calendar`

---

### Step 4. config.yaml에서 켜기

하드 제약이면:

```yaml
constraints:
  categories:
    machine: true

  hard_enabled:
    heavy_plate_block: true
```

여기서 중요한 것은 2가지입니다.

1. category가 켜져 있어야 함
2. hard_enabled 또는 soft_enabled가 켜져 있어야 함

---

## 5. 소프트 제약 추가 예시

이번에는 소프트 제약 예시입니다.

조건:
- 가능하면 `laser`를 우선 사용하고 싶다
- 하지만 반드시 그래야 하는 것은 아니다

```python
from .base import ConstraintContext, ConstraintResult


def check_prefer_laser(context: ConstraintContext) -> ConstraintResult:
    """가능하면 레이저를 우선 쓰도록 하는 소프트 제약 예시."""

    if context.machine.machine_type != "laser":
        return ConstraintResult(
            "prefer_laser",
            False,
            "non-laser selected",
            penalty=1.0,
        )

    return ConstraintResult("prefer_laser", True, "")
```

registry 등록:

```python
RULES = {
    ...
    "prefer_laser": check_prefer_laser,
}

RULE_CATEGORIES = {
    ...
    "prefer_laser": "preference",
}
```

config:

```yaml
constraints:
  categories:
    preference: true

  soft_enabled:
    prefer_laser: true

  soft_penalty_weights:
    prefer_laser: 2.0
```

이 의미는:

- 기본 penalty는 `1.0`
- 최종 적용 penalty는 `1.0 * 2.0 = 2.0`

입니다.

---

## 6. `ConstraintContext`에서 자주 쓰는 값

제약 함수 안에서 가장 자주 참조하는 값은 아래입니다.

### 작업 정보

- `context.job.job_id`
- `context.job.family`
- `context.job.thickness`
- `context.job.plate_length`
- `context.job.downstream_bay`
- `context.job.priority_weight`

### 설비 정보

- `context.machine.machine_id`
- `context.machine.machine_type`
- `context.machine.enabled`
- `context.machine.min_thickness`
- `context.machine.max_thickness`
- `context.machine.table_length_limit`
- `context.machine.daily_capacity_minutes`

### 현재 상태 정보

- `context.state.current_time`
- `context.state.machine_loads`
- `context.state.downstream_loads`
- `context.state.unscheduled_jobs`

### 날짜/운영 정보

- `context.current_day_key`
- `context.machine_enabled_flag`
- `context.machine_daily_capacity_limit`
- `context.machine_daily_load`
- `context.scheduled_job_count_today`
- `context.daily_job_cap`
- `context.global_calendar_open`
- `context.machine_calendar_open`
- `context.machine_breakdown_active`

---

## 7. 초보자용 판단 기준

새 제약을 만들 때 가장 먼저 아래를 생각하면 됩니다.

### 질문 1. 이 규칙을 어기면 "절대 안 되는가?"

- 예 → 하드 제약
- 아니오 → 소프트 제약 가능

예:
- 설비가 이 두께를 못 자름 → 하드
- 가능하면 레이저를 우선 쓰고 싶음 → 소프트

---

### 질문 2. 이 규칙은 어느 종류인가?

대충 아래처럼 보면 됩니다.

- 설비 능력/가능 여부 → `machine`
- 일일 용량/시간/건수 → `capacity`
- 후공정/적치 → `downstream`
- 납기/우선순위 → `priority`
- 선호/부하평준화 → `preference`
- 휴무/점심/고장 → `calendar`

---

### 질문 3. 날짜별 예외가 필요한가?

필요하면 hard/soft로 만들기 전에 `override`로 해결 가능한지 먼저 봅니다.

예:
- 특정 날짜 레이저 1호기 비가동
- 특정 날짜 하루 작업수 30개 제한
- 특정 날짜 설비 용량 50% 감소

이런 것은 새 제약 함수보다 override가 더 자연스럽습니다.

---

## 8. 새 제약을 추가한 뒤 확인할 것

최소한 아래 4개는 확인하는 것이 좋습니다.

### 8.1 문법 확인

```bash
python3 -m py_compile Environment/*.py Environment/constraints/*.py main.py
```

---

### 8.2 config 요약 확인

```bash
python3 main.py show-config --config config.yaml
```

여기서 새 규칙이 `hard_constraints` 또는 `soft_constraints`에 보이는지 확인합니다.

---

### 8.3 기본 시뮬레이션 확인

```bash
python3 main.py simulate --config config.yaml
```

오류 없이 끝나면 구조는 정상입니다.

---

### 8.4 step trace 확인

```bash
python3 main.py trace --config config.yaml
```

이걸 보면
- 어떤 action이 후보가 되는지
- reward가 어떻게 나오는지
- 후보가 0개일 때 시간이 어떻게 전진하는지

를 볼 수 있습니다.

---

## 9. 가장 흔한 실수

### 실수 1. 함수만 만들고 registry에 등록 안 함

이 경우:
- 코드에 함수는 있지만
- 시스템은 그 규칙을 전혀 모릅니다.

---

### 실수 2. registry에는 등록했는데 config에서 안 켬

이 경우:
- 규칙은 존재하지만 실제로는 실행되지 않습니다.

---

### 실수 3. category가 꺼져 있음

예를 들어:

```yaml
constraints:
  categories:
    machine: false
```

이면 `machine` 카테고리의 규칙은 전부 꺼집니다.

---

### 실수 4. 소프트 제약인데 penalty를 안 줌

이 경우:
- 위반은 감지되지만
- 사실상 점수에 영향이 거의 없을 수 있습니다.

---

### 실수 5. 하드로 만들었는데 너무 엄격해서 후보가 전부 사라짐

이 경우:
- 시뮬레이션이 멈추거나
- 계속 시간만 전진할 수 있습니다.

처음에는 애매한 제약을 soft로 시작하는 편이 안전합니다.

---

## 10. 추천 개발 순서

초보자 기준으로는 아래 순서를 추천합니다.

1. 먼저 규칙을 soft로 만든다
2. 결과를 trace로 본다
3. 정말 절대 규칙이라고 확신이 들면 hard로 바꾼다
4. 날짜별 예외는 가능하면 override로 분리한다

---

## 11. 체크리스트

새 제약을 추가할 때 아래 체크리스트를 따라가면 됩니다.

- [ ] 이 규칙은 hard인지 soft인지 결정했는가?
- [ ] 어느 category인지 정했는가?
- [ ] 함수 이름과 `ConstraintResult.rule_name`이 같은가?
- [ ] `registry.py`의 `RULES`에 등록했는가?
- [ ] `RULE_CATEGORIES`에 등록했는가?
- [ ] `config.yaml`에서 켰는가?
- [ ] `show-config`에서 보이는가?
- [ ] `simulate`가 오류 없이 돌아가는가?
- [ ] `trace` 결과가 의도대로 나오는가?

---

## 12. 결론

이 프로젝트에서 제약 추가는 복잡하지 않습니다.

정말로 필요한 것은 아래 3개뿐입니다.

1. 규칙 함수 1개
2. registry 등록 2줄
3. config 설정 1줄 이상

즉, 초보자도 **함수 단위로 제약을 계속 추가할 수 있는 구조**입니다.
