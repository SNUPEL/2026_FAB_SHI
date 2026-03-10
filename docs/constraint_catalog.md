# constraint catalog

이 문서는 절단 공정 PMSP 환경에서 제약조건을 어떻게 분류하고,
어떻게 코드에 추가하고, 어떻게 config로 제어할지를 정리한 문서입니다.

중요한 점은 이 프로젝트가 **단순히 hard / soft 두 분류만 쓰는 구조가 아니라**,
아래 3개 축을 동시에 보는 구조로 가는 것이 좋다는 점입니다.

1. `제약의 종류(category)`
2. `제약의 강도(hard / soft)`
3. `제약의 적용 방식(mask / penalty / override / relax)`

판넬라인 프로젝트도 실제로는 이 3개 축을 함께 썼습니다.
예를 들어 판넬라인에서는:

- `enabled_constraints_*`: 어떤 제약을 검사할지
- `hard_constraints_*`, `strict_rules`: 절대 완화 금지
- `relax_order_*`: 완화 순서
- `calendar`: 점심 / 휴무 / 오전중지 / 오후중지
- `daily_block_cap_overrides`, `daily_seam_cap_overrides`: 일별 제약 오버라이드

처럼, 단순 hard / soft가 아니라 **종류 + 운영 설정 + 완화 방식**을 함께 관리했습니다.

현재 `pmsp`도 같은 철학으로 가는 것이 맞습니다.

---

## 1. 현재 pmsp 코드 기준 분류

현재 코드는 아래 4개를 실제로 사용합니다.

- `categories`
- `hard_enabled`
- `soft_enabled`
- `overrides`

즉,

- category: 큰 분류 전체 on/off
- hard 제약: action masking으로 바로 제거
- soft 제약: 선택은 가능하지만 penalty 부여
- override: 날짜별 / 설비별 예외 운영 반영

---

## 2. 추천 분류 체계

절단 공정 기준으로는 아래처럼 분류하는 것이 가장 직관적입니다.

### 2.1 설비 제약

설비 자체의 능력 / 사용 가능 여부와 직접 관련된 제약입니다.

예:
- 설비 사용 가능 여부
- 설비 타입별 처리 가능 계열
- 설비 타입별 처리 가능 두께
- 정반 길이 제한
- 설비별 단일 작업 처리
- 설비별 속도 차이

현재 코드 키 예:
- `machine_enabled`
- `family_eligibility`
- `thickness_range`
- `table_length_limit`
- `machine_single_processing`

권장 강도:
- 대부분 hard

---

### 2.2 용량 / 시간 제약

시간과 생산능력, 하루 운영 한계와 관련된 제약입니다.

예:
- 설비별 일일 가용시간
- 일일 총 투입 작업 수
- 일일 절단 총 길이 / 면적 / 중량 한계
- 셋업 포함 총 가동시간 한계
- 작업 시작 가능 시각 / 마감 시각

현재 코드 키 예:
- `daily_capacity_limit`

향후 추가 후보:
- `daily_job_count_limit`
- `daily_cut_length_limit`
- `shift_time_limit`
- `release_time_limit`

권장 강도:
- 일부는 hard
- 일부는 soft 또는 override

예:
- “절대 1일 16시간 이상 운전 불가” → hard
- “가능하면 1일 12시간 이하 선호” → soft

---

### 2.3 후공정 / 적치 제약

절단 이후 베이 또는 후공정 작업장과의 연계를 보는 제약입니다.

예:
- 특정 bay 적치 초과 금지
- bay 우선순위 반영
- bay 혼잡 경고
- 특정 bay는 특정 설비와의 연결 선호

현재 코드 키 예:
- `downstream_capacity`
- `downstream_priority`
- `downstream_buffer_warning`

권장 강도:
- 적치 초과는 hard
- 우선순위나 혼잡 경고는 soft

---

### 2.4 우선순위 / 납기 제약

작업의 긴급도와 운영 우선순위에 관련된 제약입니다.

예:
- 납기 여유 부족 작업 우선
- 긴급 작업 우선
- 특정 후공정 연결 작업 우선
- 특정 고객 / 계열 우선

현재 코드 키 예:
- `due_date_urgency`

향후 추가 후보:
- `rush_job_priority`
- `customer_priority`
- `due_date_hard_limit`

권장 강도:
- 보통 soft
- 단, “지연 금지”는 hard 가능

---

### 2.5 부하 평준화 제약

설비 간 workload를 가능한 한 균등하게 배분하려는 제약입니다.

예:
- 특정 설비 과부하 방지
- 레이저 / 플라즈마 간 사용량 균형
- 설비군별 배정 편향 완화

현재 코드 키 예:
- `load_balance_preference`

권장 강도:
- 대부분 soft

---

### 2.6 선호 / 정책성 제약

절대 규칙은 아니지만, 운영자가 원할 수 있는 정책성 규칙입니다.

예:
- 얇은 재질은 레이저 우선
- 특정 계열은 플라즈마 우선
- 특정 설비는 야간에만 사용

현재 코드 키 예:
- `preferred_machine_type`

권장 강도:
- soft

---

### 2.7 물리 / 레이아웃 제약

아직 데이터가 없어서 실제 코드에는 안 들어갔지만,
향후 2D 환경이나 설비 배치 변경까지 고려하면 필요한 분류입니다.

예:
- 특정 설비 구역에서 특정 크기 이상의 정반 불가
- 이동 거리 / 물류 경로 제약
- 레이저 신규 도입 위치 기반 연결 제약

향후 추가 후보:
- `layout_zone_limit`
- `transport_distance_limit`
- `machine_cluster_preference`

권장 강도:
- hard 또는 soft 둘 다 가능

---

### 2.8 캘린더 / 운영일 제약

판넬라인에서 매우 중요했던 축입니다.
절단 공정에도 실제로 들어올 가능성이 높습니다.

예:
- 점심시간 중지
- 휴무일
- 특정 날짜 라인 축소 가동
- 특정 날짜는 레이저 2대만 운용
- 특정 날짜는 용량 축소 / 확대
- 특정 날짜 / 특정 시간은 설비 고장으로 비가동

현재 pmsp 코드에는 아래 규칙이 실제로 들어가 있습니다.

- `calendar_open`
  - 휴무일 / 점심시간 / 공장 전체 shutdown window
- `machine_calendar_open`
  - 설비별 운영시간 / 설비별 계획 정지
- `machine_breakdown`
  - 특정 설비의 시간대별 고장 / 정지

중요:
- 현재는 `작업 시작 시점` 기준입니다.
- 작업 도중 고장/점심시간이 끼어드는 preemption은 아직 미구현입니다.

향후 추가 후보:
- `enable_holidays_off`
- `enable_half_day_off`
- `enable_lunch_break`
- `enable_global_shutdown_windows`
- `enable_machine_operating_windows`
- `enable_machine_shutdown_windows`
- `enable_machine_breakdowns`
- `daily_machine_capacity_overrides`
- `daily_machine_enable_overrides`
- `daily_job_cap_overrides`

권장 강도:
- 운영 오버라이드 성격

---

## 3. hard / soft / override / relax

같은 category라도 적용 방식은 다를 수 있습니다.

### 3.1 Hard

위반하면 action 후보에서 제거합니다.

예:
- 두께 범위 위반
- 정반 길이 초과
- 적치 초과
- 설비 비활성

현재 구현 방식:
- `constraints.hard_enabled`

---

### 3.2 Soft

선택은 가능하지만 penalty를 줍니다.

예:
- 우선순위가 더 높은 bay가 남아 있음
- 납기 여유가 부족한데 늦은 배정
- 이미 과부하된 설비 사용

현재 구현 방식:
- `constraints.soft_enabled`
- `constraints.soft_penalty_weights`

---

### 3.3 Override

특정 날짜 / 특정 설비 / 특정 조건에서 기본 제약값을 덮어씁니다.

판넬라인에서 실제로 중요했던 방식입니다.

예:
- `2026-07-10`은 레이저 1호기 용량 50%만 사용
- `2026-07-15`는 플라즈마 3호기 비가동
- 특정 날짜는 일일 작업수 제한 30건

판넬라인 예:
- `daily_block_cap_overrides`
- `daily_seam_cap_overrides`
- `daily_seam_cap_scales`

pmsp에서도 이 방식은 매우 유용합니다.

---

### 3.4 Relax

후보가 하나도 없을 때, 일부 제약을 순서대로 풀어주는 방식입니다.

이건 현재 `pmsp` 코드에는 아직 없습니다.
하지만 판넬라인에서는 핵심 기능이었습니다.

판넬라인 예:
- `relax_order_start_date`
- `relax_order_assembly`
- `strict_rules`

pmsp에서도 나중에 아래처럼 확장 가능성이 있습니다.

예:
1. 선호 설비 타입 완화
2. 부하 평준화 제약 완화
3. bay 우선순위 제약 완화
4. 일일 soft 한계 완화

다만 초반에는 hard / soft / override까지만 두는 편이 안전합니다.

---

## 4. 현재 코드에 이미 들어간 제약

### 4.1 설비 제약

- `machine_enabled`
- `family_eligibility`
- `thickness_range`
- `table_length_limit`
- `machine_single_processing`

파일:
- [Environment/constraints/machine_rules.py](../Environment/constraints/machine_rules.py)

---

### 4.2 용량 제약

- `daily_capacity_limit`
- `daily_job_cap_limit`

파일:
- [Environment/constraints/machine_rules.py](../Environment/constraints/machine_rules.py)

---

### 4.3 후공정 / 적치 제약

- `downstream_capacity`
- `downstream_priority`
- `downstream_buffer_warning`

파일:
- [Environment/constraints/downstream_rules.py](../Environment/constraints/downstream_rules.py)

---

### 4.4 우선순위 / 선호 / 평준화 제약

- `due_date_urgency`
- `preferred_machine_type`
- `load_balance_preference`

파일:
- [Environment/constraints/soft_rules.py](../Environment/constraints/soft_rules.py)

---

### 4.5 캘린더 / 설비 고장 제약

- `calendar_open`
- `machine_calendar_open`
- `machine_breakdown`

파일:
- [Environment/constraints/calendar_rules.py](../Environment/constraints/calendar_rules.py)

---

## 5. 초보자용 제약 추가 절차

이 구조는 초보자가 함수 단위로 제약을 쉽게 추가할 수 있게 만든 것입니다.

### Step 1. 규칙 함수 1개 작성

예를 들어 새 규칙 `check_machine_cluster_limit`를 만든다고 가정합니다.

```python
def check_machine_cluster_limit(context: ConstraintContext) -> ConstraintResult:
    passed = True
    reason = ""
    penalty = 0.0
    return ConstraintResult("machine_cluster_limit", passed, reason, penalty=penalty)
```

작성 위치:
- 설비 규칙이면 `machine_rules.py`
- 후공정 규칙이면 `downstream_rules.py`
- 점수성 규칙이면 `soft_rules.py`

---

### Step 2. registry에 이름 등록

[Environment/constraints/registry.py](../Environment/constraints/registry.py)의 `RULES`에 추가합니다.

```python
"machine_cluster_limit": check_machine_cluster_limit,
```

---

### Step 3. config.yaml에서 켜기

하드 규칙이면:

```yaml
constraints:
  hard_enabled:
    machine_cluster_limit: true
```

소프트 규칙이면:

```yaml
constraints:
  soft_enabled:
    machine_cluster_limit: true
  soft_penalty_weights:
    machine_cluster_limit: 1.0
```

---

### Step 4. 끝

simulation 코드는 따로 건드릴 필요가 없습니다.
후보 평가 시 자동으로 적용됩니다.

---

## 6. 판넬라인 프로젝트에서 배울 점

판넬라인은 제약을 아래처럼 다뤘습니다.

### 6.1 어떤 제약을 검사할지

- `enabled_constraints_*`

즉, **검사 대상 목록**을 따로 뒀습니다.

---

### 6.2 어떤 제약을 절대 하드로 둘지

- `hard_constraints_*`
- `strict_rules`

즉, **완화 금지 목록**을 따로 뒀습니다.

---

### 6.3 어떤 순서로 제약을 풀지

- `relax_order_*`

즉, **후보가 없을 때의 운영 순서**를 따로 뒀습니다.

---

### 6.4 날짜별 운영 오버라이드

- `calendar`
- `daily_block_cap_overrides`
- `daily_seam_cap_overrides`
- `daily_seam_cap_scales`

즉, **기본 규칙과 예외 운영을 분리**했습니다.

---

## 7. pmsp도 같은 방향으로 가는 것이 좋은가?

네, 좋습니다.

현재는 아래가 구현된 상태입니다.

- `categories`
- `hard_enabled`
- `soft_enabled`
- `overrides`
- `calendar` 하드 제약

하지만 이후에는 아래처럼 확장하는 것이 가장 자연스럽습니다.

### 추천 확장 방향

1. `categories`
   - machine
   - capacity
   - downstream
   - priority
   - preference
   - layout
   - calendar

2. `hard_enabled`
3. `soft_enabled`
4. `override`
5. `relax_order` (필요 시)

즉,
**종류별 분류 + hard/soft + 운영 오버라이드** 체계가 가장 안정적입니다.

---

## 8. 결론

현재 pmsp는:

- hard / soft는 이미 구현됨
- category 분류도 코드와 문서에 반영됨
- override는 코드에 기본 구현됨
- calendar 제약과 설비 고장도 구현됨
- relax는 아직 미구현이지만, 판넬라인처럼 확장 가능

따라서 지금 단계의 가장 좋은 운영 원칙은 이렇습니다.

1. 먼저 제약을 `종류(category)`로 정리한다.
2. 각 제약을 hard 또는 soft로 결정한다.
3. 날짜별 / 설비별 예외는 override로 분리한다.
4. 정말 필요할 때만 relax_order를 넣는다.
