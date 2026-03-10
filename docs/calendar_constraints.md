# calendar constraints

이 문서는 절단 공정 PMSP 환경에서 `calendar` 계열 제약을 어떻게 쓰는지 정리합니다.

핵심 아이디어는 단순합니다.

- 어떤 작업이 **지금 시작 가능한지**를 calendar가 결정합니다.
- 공장 전체 휴무, 점심시간, shutdown window, 설비 고장은 모두
  "현재 시각에 새 작업을 시작할 수 있는가?"로 해석합니다.

중요:
- 현재 구현은 **작업 시작 시점 기준**입니다.
- 즉, 이미 시작한 작업이 점심시간이나 고장 시간에 걸쳐도
  중간 정지했다가 다시 이어서 하는 `preemption / resume`은 아직 구현하지 않았습니다.

---

## 1. 현재 구현된 calendar 제약

현재 코드에 실제로 들어가 있는 규칙은 아래 3개입니다.

### 1.1 `calendar_open`

공장 전체가 현재 시각에 가동 가능한지 봅니다.

이 규칙 안에서 확인하는 것:
- `holidays_off`
- `lunch_break`
- `global_shutdown_windows`

즉, 설비 개별 문제가 아니라 **공장 전체 운영 상태**를 나타냅니다.

코드:
- [Environment/constraints/calendar_rules.py](../Environment/constraints/calendar_rules.py)
- [Environment/simulation.py](../Environment/simulation.py)

---

### 1.2 `machine_breakdown`

특정 설비가 현재 시각에 고장/정지 상태인지 봅니다.

예:
- 레이저 1호기 오전 고장
- 플라즈마 2호기 종일 정지

이 규칙은 설비 단위입니다.

코드:
- [Environment/constraints/calendar_rules.py](../Environment/constraints/calendar_rules.py)
- [Environment/simulation.py](../Environment/simulation.py)

---

### 1.3 `machine_calendar_open`

특정 설비가 현재 시각에 "계획된 운영시간 안"에 있는지 봅니다.

이 규칙이 필요한 이유:
- 고장은 비계획 정지입니다.
- 하지만 현업에서는 설비마다 계획된 가동시간이 다를 수 있습니다.
  - 레이저는 주간만
  - 플라즈마는 24시간
  - 특정 설비는 오전만

즉, `machine_breakdown`과 `machine_calendar_open`은 분리하는 것이 맞습니다.

코드:
- [Environment/constraints/calendar_rules.py](../Environment/constraints/calendar_rules.py)
- [Environment/simulation.py](../Environment/simulation.py)

---

## 2. config.yaml에서 켜고 끄는 방법

calendar category 전체를 켜려면:

```yaml
constraints:
  categories:
    calendar: true
```

하드 제약으로 실제 마스킹하려면:

```yaml
constraints:
  hard_enabled:
    calendar_open: true
    machine_calendar_open: true
    machine_breakdown: true
```

즉:
- `categories.calendar = false`
  - calendar 규칙 전체 꺼짐
- `categories.calendar = true` + `hard_enabled.calendar_open = true`
  - 공장 전체 휴무/점심/shutdown 적용
- `categories.calendar = true` + `hard_enabled.machine_calendar_open = true`
  - 설비별 운영시간 / 계획 정지 적용
- `categories.calendar = true` + `hard_enabled.machine_breakdown = true`
  - 설비 고장 적용

---

## 3. 지원하는 calendar 설정

### 3.1 휴무일

```yaml
calendar:
  enable_holidays_off: true
  holidays_off:
    - "2026-01-01"
    - "2026-01-10~2026-01-12"
    - "20260120..20260122"
```

지원 형식:
- `YYYY-MM-DD`
- `YYYYMMDD`
- `A~B`
- `A..B`

동작:
- 해당 날짜에는 어떤 설비도 새 작업을 시작하지 않습니다.

---

### 3.1-1 반일 가동

```yaml
calendar:
  enable_half_day_off: true
  half_day_off:
    - "2026-01-05"
  half_day_off_window: "12:00-24:00"
```

의미:
- 해당 날짜의 지정 시간대에는 공장 전체 새 작업 시작 금지
- 전형적인 예시는 "오후 반일 휴무"입니다.

---

### 3.2 점심시간

```yaml
calendar:
  enable_lunch_break: true
  lunch_break: "12:00-13:00"
  lunch_break_dates: []
```

의미:
- `lunch_break_dates`가 비어 있으면 모든 날짜에 적용
- 날짜 목록이 있으면 해당 날짜에만 적용

예:

```yaml
calendar:
  enable_lunch_break: true
  lunch_break: "11:30-12:30"
  lunch_break_dates:
    - "2026-01-03"
    - "2026-01-04"
```

---

### 3.3 공장 전체 shutdown window

```yaml
calendar:
  enable_global_shutdown_windows: true
  global_shutdown_windows:
    "2026-01-02":
      - "08:00-10:00"
      - "15:00-16:00"
```

의미:
- 특정 날짜 특정 시간대에 공장 전체 새 작업 시작 금지

활용 예:
- 정기 점검
- 일시 전원 차단
- 공장 전체 시험 운전

---

### 3.4 설비 고장 / 설비 정지

```yaml
calendar:
  enable_machine_breakdowns: true
  machine_breakdowns:
    "2026-01-01":
      laser_01:
        - "09:00-12:00"
      plasma_02: full_day
```

지원 형식:
- 시간 구간 list
- `full_day`
- `true`

예:

```yaml
calendar:
  enable_machine_breakdowns: true
  machine_breakdowns:
    "2026-01-01":
      laser_01:
        - "09:00-12:00"
        - "14:00-15:30"
      plasma_01: full_day
      plasma_02: true
```

의미:
- `laser_01`은 두 번 멈춤
- `plasma_01`, `plasma_02`는 종일 비가동

---

### 3.5 설비별 운영시간

```yaml
calendar:
  enable_machine_operating_windows: true
  machine_operating_windows:
    default:
      laser_01:
        - "08:00-18:00"
      plasma_01:
        - "00:00-24:00"
    "2026-01-02":
      laser_01:
        - "10:00-16:00"
```

의미:
- `default`는 기본 운영시간
- 특정 날짜 key가 있으면 그 날짜 설정이 우선

즉, 같은 날짜라도 설비마다 다르게 설정할 수 있습니다.

예:
- `laser_01`: 주간만 가동
- `plasma_01`: 24시간 가동

---

### 3.6 설비별 계획 정지

```yaml
calendar:
  enable_machine_shutdown_windows: true
  machine_shutdown_windows:
    "2026-01-01":
      laser_01:
        - "13:00-14:00"
      plasma_02:
        - "09:00-10:30"
```

의미:
- 고장과 달리, 계획된 정지/비가동입니다.
- 예: 청소, 점검, 교정, 작업자 부재

---

## 4. 자동 시간 전진은 어떻게 동작하나

calendar 제약을 넣으면 가장 중요한 문제가 생깁니다.

- 지금 시각에는 후보가 0개
- 하지만 시간이 조금만 지나면 다시 후보가 생김

예:
- 현재 12:10이고 점심시간이라 후보 0개
- 현재 09:30이고 `laser_01`이 고장이라 후보 0개
- 현재 오늘 작업 수 cap에 걸려서 내일은 다시 가능

그래서 현재 시뮬레이션은 후보가 없을 때:

1. 설비가 바빠서 후보가 없는 경우
   - 다음 설비 완료 시점으로 점프

2. calendar / 고장 / 일일 용량 때문에 후보가 없는 경우
   - 1분씩 전진해서 다시 후보를 찾음

이렇게 동작합니다.

장점:
- 초보자가 이해하기 쉽습니다.
- 점심시간 / 고장 / 휴무 같은 규칙이 직관적으로 먹습니다.

단점:
- 매우 긴 shutdown이 많으면 느릴 수 있습니다.

현재 단계에서는 이 방식이 가장 안전합니다.

---

## 5. 초보자 기준으로 설비 고장을 추가하는 방법

코드 수정 없이도 대부분은 config만 바꾸면 됩니다.

예:

```yaml
constraints:
  categories:
    calendar: true
  hard_enabled:
    machine_breakdown: true

calendar:
  enable_machine_breakdowns: true
  machine_breakdowns:
    "2026-01-01":
      laser_01:
        - "09:00-12:00"
```

이렇게만 하면,
- 2026-01-01 09:00~12:00 동안 `laser_01`은 후보에서 제거됩니다.

---

## 6. 현재 한계

현재는 아직 아래 기능이 없습니다.

1. 작업 중간 고장
- 작업이 08:50에 시작해서 09:00에 고장 나면
  지금은 그 작업을 끊지 않습니다.

2. 고장 복구 후 남은 작업시간 재계산
- 아직 없습니다.

3. 작업 중간 계획 정지 / 고장 반영
- 현재는 "시작 시점 기준"으로만 판단합니다.

4. 교체/보수에 따른 setup 증가
- 아직 없습니다.

---

## 7. 향후 확장 추천

다음 단계에서 자연스럽게 넣을 수 있는 항목:

1. 설비별 운영시간 캘린더
- 예: 레이저는 08:00-20:00만 가동

2. 설비별 정비 주기
- 예: 누적 300분마다 20분 정비

3. preemption / resume
- 작업 중간 점심시간 / 고장 반영

4. shift calendar
- 주간조 / 야간조 / 휴무조

---

## 8. 결론

현재 `pmsp`는 calendar 계열에서 아래를 이미 지원합니다.

- 휴무일
- 반일 가동
- 점심시간
- 공장 전체 shutdown window
- 설비별 운영시간
- 설비별 계획 정지
- 설비 고장 / 설비 비가동

그리고 이들은 모두:
- `constraints.categories.calendar`
- `constraints.hard_enabled.calendar_open`
- `constraints.hard_enabled.machine_calendar_open`
- `constraints.hard_enabled.machine_breakdown`
- `calendar.*`

으로 제어됩니다.
