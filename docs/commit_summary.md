# commit summary

이 문서는 현재 작업 내용을 `git commit` 또는 PR 설명에 바로 쓸 수 있게 정리한 문서입니다.

---

## 1. 추천 commit title

### 짧은 버전

```text
Implement PMSP cutting-shop skeleton with RL, constraints, and calendar controls
```

### 조금 더 설명적인 버전

```text
Build PMSP cutting-shop environment, RL trainers, modular constraint framework, and calendar controls
```

---

## 2. 추천 commit body

```text
- implement PMSP-style cutting shop environment with reset/step interface
- add job-machine pair action model and heuristic simulation flow
- add PPO, REINFORCE, and self-labeling training pipelines
- add GNN/MLP-based policy-value network structure
- add modular constraint system with category, hard, soft, and override controls
- add calendar rules for holidays, lunch break, half-day shutdown, machine operating windows, planned shutdowns, and machine breakdowns
- add beginner-friendly documentation for constraints, config, and implementation status
- add project .gitignore and improve top-level README
```

---

## 3. 추천 PR summary

아래 문구는 PR 설명이나 보고용 요약으로 쓸 수 있습니다.

### 국문 버전

```text
이번 커밋에서는 절단 공정을 PMSP형 병렬 이기종 설비 스케줄링 문제로 다룰 수 있는 기본 코드 프레임워크를 구축했다.
환경은 reset/step 기반으로 동작하며, 작업-설비 pair를 action으로 사용한다.
제약조건은 category / hard / soft / override 구조로 분리하여 초보자도 함수 단위로 쉽게 추가할 수 있게 정리했다.
또한 calendar 제약을 통해 휴무일, 점심시간, 반일 가동, 설비별 운영시간, 계획 정지, 설비 고장을 반영할 수 있게 했다.
학습 알고리즘은 PPO, REINFORCE, Self-labeling 구조를 분리 구현했고, GNN + MLP 기반 정책/가치 네트워크 골격을 연결했다.
현재 버전은 실데이터 기반 최종 모델이 아니라, 연구계획서 전체를 코드로 확장할 수 있는 실행 가능한 스켈레톤 프레임워크다.
```

### 영문 버전

```text
This commit establishes an executable PMSP-style cutting-shop scheduling framework.
The environment now provides a reset/step interface with job-machine pair actions.
Constraints are organized by category, hard/soft behavior, and date-based overrides so that new rules can be added with simple standalone functions.
Calendar controls were added for holidays, lunch breaks, half-day shutdowns, machine operating windows, planned shutdowns, and machine breakdowns.
The training stack now includes PPO, REINFORCE, and self-labeling pipelines with a GNN/MLP-based policy-value network backbone.
This is not the final production-ready model yet, but a runnable and extensible skeleton aligned with the research plan.
```

---

## 4. 아주 짧은 보고용 한 줄

```text
PMSP형 절단 스케줄링 환경, RL 학습 구조, 제약 프레임워크, calendar 제어 기능까지 기본 골격 구현 완료
```
