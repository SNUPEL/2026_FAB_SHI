# config guide

## 1. 왜 config.yaml이 중요한가

이 프로젝트는 초보자도 제약조건을 쉽게 켜고 끌 수 있게 만드는 것이 목표입니다.
그래서 핵심 제어는 `config.yaml`에서 하도록 설계했습니다.

판넬라인 프로젝트도 결국은 `config에서 운영 규칙을 바꾸는 방식`이 핵심이었습니다.

예:
- 어떤 제약을 검사할지
- 어떤 제약을 하드로 볼지
- 어떤 완화 순서를 쓸지
- 어떤 날짜에 용량을 줄이거나 늘릴지

`pmsp`도 같은 방향으로 가는 것이 맞습니다.

---

## 2. 현재 config.yaml 주요 섹션

### 2.1 paths

- `scenario_path`: 입력 시나리오 파일
- `output_dir`: 결과 저장 폴더

---

### 2.2 simulation

- `horizon_minutes`: 시뮬레이션 최대 길이
- `auto_advance_time`: 액션이 없을 때 다음 완료 시점으로 자동 전진할지 여부
- `decision_mode`: 현재는 `decision_epoch`
- `stage_names`: `setup / cut / finish`

---

### 2.3 action_space

- `mode`: 현재는 `job_machine_pair`

즉, action 1개는 “작업 1개를 설비 1개에 넣는 배정”입니다.

---

### 2.4 network

- `hidden_dim`
- `num_gnn_layers`

현재는 GNN + MLP 구조의 후보 action scoring 네트워크에 사용됩니다.

---

### 2.5 constraints.categories

- category 단위 on/off
- 여기서 `false`인 분류는 그 아래 규칙이 전부 꺼집니다.

예:
- machine
- capacity
- downstream
- priority
- preference
- layout
- calendar

---

### 2.6 constraints.hard_enabled

- 하드 제약 on/off
- 여기서 `true`인 항목은 action masking으로 바로 제거됩니다.

예:
- 두께 범위 위반
- 길이 초과
- 적치 초과
- 휴무일 / 점심시간 / 공장 전체 정지
- 설비 고장

---

### 2.7 constraints.soft_enabled

- 소프트 제약 on/off
- 여기서 `true`인 항목은 penalty로만 반영됩니다.

예:
- 우선순위 선호
- 부하 평준화 선호
- 납기 여유 선호

---

### 2.8 constraints.soft_penalty_weights

- 각 소프트 제약의 penalty 크기

즉, “어떤 소프트 제약을 얼마나 강하게 벌점 줄지”를 여기서 조절합니다.

---

### 2.9 constraints.overrides

- 날짜별 / 설비별 예외 운영
- 판넬라인의 `daily_*_overrides`와 같은 역할입니다.

현재 코드가 읽는 항목:
- `enable_daily_machine_capacity_overrides`
- `enable_daily_machine_enable_overrides`
- `enable_daily_job_cap_overrides`
- `daily_machine_capacity_overrides`
- `daily_machine_enable_overrides`
- `daily_job_cap_overrides`

---

### 2.10 calendar

- 운영일 / 점심시간 / 공장 전체 정지 / 설비 고장을 설정하는 섹션

현재 코드가 읽는 항목:
- `enable_holidays_off`
- `holidays_off`
- `enable_half_day_off`
- `half_day_off`
- `half_day_off_window`
- `enable_lunch_break`
- `lunch_break`
- `lunch_break_dates`
- `enable_global_shutdown_windows`
- `global_shutdown_windows`
- `enable_machine_operating_windows`
- `machine_operating_windows`
- `enable_machine_shutdown_windows`
- `machine_shutdown_windows`
- `enable_machine_breakdowns`
- `machine_breakdowns`
- `enable_operation_time_adjustment`
- `operation_time_adjustment_limit_minutes`

핵심 규칙:
- `constraints.hard_enabled.calendar_open`
- `constraints.hard_enabled.machine_calendar_open`
- `constraints.hard_enabled.machine_breakdown`

중요:
- 기본 hard mask는 여전히 `작업 시작 시점` 기준입니다.
- 다만 `enable_operation_time_adjustment: true`이면 작업 구간 중 비가동 시간만큼 완료시각을 뒤로 미룹니다.

---

### 2.11 setup

- `enable_family_changeover`
- `default_family_changeover_minutes`
- `machine_type_changeover_minutes`

즉, 직전 작업과 계열이 바뀌면 추가 셋업 시간을 반영할 수 있습니다.

---

### 2.12 reward

- `makespan_weight`
- `load_balance_weight`
- `priority_bonus_weight`
- `soft_penalty_weight`

즉, soft 제약 penalty 자체와 별도로
reward에서 어떤 목표를 얼마나 중요하게 볼지도 설정합니다.

---

### 2.13 train

- `episodes`
- `algorithm`

현재 선택 가능:
- `ppo`
- `reinforce`
- `self_labeling`

---

## 3. 현재 구조의 장점

현재 구조는 단순합니다.

- 제약 함수는 코드에 1개 추가
- config에서 `true / false`만 바꾸면 됨

초보자 입장에서는 이 방식이 가장 이해하기 쉽습니다.

---

## 4. 하지만 앞으로는 "종류별 분류"도 같이 두는 것이 좋다

현재는 실제로 아래 네 축과 calendar 섹션을 함께 씁니다.

- `categories`
- `hard_enabled`
- `soft_enabled`
- `overrides`
- `calendar`

하지만 실제 운영에서는 이것만으로는 부족할 수 있습니다.

왜냐하면 사람들은 보통 제약을 이렇게 생각하기 때문입니다.

- 설비 제약
- 용량 제약
- 후공정 제약
- 우선순위 제약
- 캘린더 제약

즉, **사람이 이해하는 분류(category)**와
**시스템이 적용하는 강도(hard / soft)**를 같이 두는 것이 더 좋습니다.

---

## 5. 추천하는 장기 config 구조

현재 코드를 당장 다 바꾸자는 뜻은 아니고,
향후 확장 시 아래 구조가 가장 자연스럽다는 뜻입니다.

```yaml
constraints:
  categories:
    machine: true
    capacity: true
    downstream: true
    priority: true
    preference: true
    layout: false
    calendar: false

  hard_enabled:
    machine_enabled: true
    family_eligibility: true
    thickness_range: true
    table_length_limit: true
    machine_single_processing: true
    daily_capacity_limit: true
    daily_job_cap_limit: false
    downstream_capacity: true

  soft_enabled:
    downstream_priority: true
    due_date_urgency: true
    preferred_machine_type: true
    downstream_buffer_warning: true
    load_balance_preference: true

  soft_penalty_weights:
    downstream_priority: 3.0
    due_date_urgency: 2.0
    preferred_machine_type: 1.0
    downstream_buffer_warning: 1.5
    load_balance_preference: 0.5

  overrides:
    enable_daily_machine_capacity_overrides: false
    enable_daily_machine_enable_overrides: false
    enable_daily_job_cap_overrides: false
```

이 구조의 장점:

1. 초보자는 category로 큰 틀을 이해할 수 있음
2. 개발자는 hard / soft로 실제 동작을 제어할 수 있음
3. 운영자는 override로 날짜별 예외를 줄 수 있음

---

## 6. 판넬라인에서는 일별 제약을 어떻게 정리했는가

판넬라인은 아래처럼 분리했습니다.

### 6.1 calendar

- 점심시간
- 하루 휴무
- 오전 중지
- 오후 중지

즉, 시간대 운영 자체를 제어했습니다.

---

### 6.2 daily overrides

- `daily_block_cap_overrides`
- `daily_seam_cap_overrides`
- `daily_seam_cap_scales`

즉, 날짜별로 기본 제약값을 덮어썼습니다.

예:
- 어떤 날은 블록 수 15개만 가능
- 어떤 날은 심수 용량 90으로 제한
- 어떤 날은 심수 용량 1.2배로 확대

---

### 6.3 hard / strict / relax

- `hard_constraints_*`
- `strict_rules`
- `relax_order_*`

즉,
- 절대 풀면 안 되는 것
- 후보가 없을 때 순서대로 푸는 것

을 분리했습니다.

---

## 7. pmsp에서 일별 제약을 넣는다면 어떻게 가는 게 좋은가

절단 공정 기준으로는 아래가 자연스럽습니다.

### 7.1 daily machine capacity override

예:
- 특정 날짜는 레이저 1호기 50% 용량
- 특정 날짜는 플라즈마 2호기 full 가동

예시:

```yaml
constraints:
  overrides:
    enable_daily_machine_capacity_overrides: true
    daily_machine_capacity_overrides:
      "2026-07-10":
        LZ-01: 300
        PL-02: 480
```

---

### 7.2 daily machine enable override

예:
- 특정 날짜는 설비 비가동
- 특정 날짜는 신규 레이저만 사용

```yaml
constraints:
  overrides:
    enable_daily_machine_enable_overrides: true
    daily_machine_enable_overrides:
      "2026-07-15":
        LZ-03: false
        PL-01: true
```

---

### 7.3 daily job cap override

예:
- 특정 날짜 총 작업 수 제한
- 특정 날짜 총 절단 길이 제한

```yaml
constraints:
  overrides:
    enable_daily_job_cap_overrides: true
    daily_job_cap_overrides:
      "2026-07-20": 30
```

---

### 7.4 calendar

예:
- 휴무일
- 반일 가동
- 점심시간

절단 공정에서도 실제로는 충분히 필요할 수 있습니다.

---

## 8. 가장 자주 바꾸게 될 부분

### 8.1 하드 제약 끄기 예시

```yaml
constraints:
  hard_enabled:
    table_length_limit: false
```

---

### 8.2 소프트 제약 켜기 예시

```yaml
constraints:
  soft_enabled:
    downstream_priority: true
```

---

### 8.3 penalty 크기 조정 예시

```yaml
constraints:
  soft_penalty_weights:
    downstream_priority: 5.0
```

---

### 8.4 향후 override 추가 예시

```yaml
constraints:
  overrides:
    enable_daily_machine_capacity_overrides: true
    daily_machine_capacity_overrides:
      "2026-08-01":
        LZ-01: 240
```

---

## 9. 운영 원칙

1. 절대 불가능한 규칙은 hard
2. 가능하지만 피하고 싶은 규칙은 soft
3. 날짜별 예외 운영은 override로 분리
4. 사람이 이해하는 구조를 위해 category 문서를 함께 유지
5. 애매한 규칙은 처음에는 soft로 시작하는 편이 안전

---

## 10. 결론

현재 코드의 실제 동작은 아래 다섯 축입니다.

- `categories`
- `hard_enabled`
- `soft_enabled`
- `overrides`
- `calendar`

하지만 문서와 설계 기준으로는:

- `종류(category)`
- `강도(hard / soft)`
- `운영 예외(override)`

세 축으로 보는 것이 가장 좋습니다.

이 방향은 판넬라인 프로젝트에서 이미 검증된 방식과도 잘 맞습니다.
