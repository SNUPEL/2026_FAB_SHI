# RL References

이 프로젝트의 현재 구현은 아래 논문/코드의 아이디어를 단순화해서 반영했습니다.

## 1. PPO / GNN 기반 스케줄링

- Junyoung Park et al., "Learning to schedule job-shop problems: Representation and policy learning using graph neural network and reinforcement learning", 2021
  - 링크: https://arxiv.org/abs/2106.01086
  - 반영한 부분:
    - 상태를 그래프/후보 집합으로 보고 representation + policy를 분리하는 관점
    - PPO 기반 정책 학습

- Bulent Soykan et al., "Graph-Enhanced Deep Reinforcement Learning for Multi-Objective Unrelated Parallel Machine Scheduling", 2026
  - 링크: https://arxiv.org/abs/2602.08052
  - 반영한 부분:
    - UPMSP 계열 문제를 GNN + PPO로 푸는 방향
    - 설비 이질성, eligibility, multi-objective reward 관점

## 2. 코드 참고: L2D

- GitHub: https://github.com/zcaicaros/L2D
  - 논문: "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" (NeurIPS 2020)
  - 반영한 부분:
    - action 후보를 순차적으로 선택하는 dispatching 스타일
    - 학습 환경과 정책 네트워크를 분리하는 구조

## 3. Self-labeling 참고

- GitHub: https://github.com/AndreaCorsini1/SelfLabelingJobShop
  - 저장소 설명: "Self-Labeling the Job Shop Scheduling Problem"
  - 반영한 부분:
    - 여러 rollout을 샘플링
    - 가장 좋은 rollout을 pseudo-label로 삼아 imitation update 수행

## 4. 현재 코드와 차이

현재 pmsp 코드의 구현은 "연구용 간소 버전"입니다.

- 논문/공개코드 대비 단순화한 부분
  - PyG/DGL 같은 외부 GNN 프레임워크 미사용
  - message passing을 adjacency 기반 평균 집계로 단순화
  - PPO를 mini-batch GAE 버전이 아닌 episode-level 간단 버전으로 구현
  - self-labeling도 best rollout imitation 형태로 최소 구현

- 현재 코드의 목적
  - 절단 공정용 PMSP 환경에 바로 붙일 수 있는 최소 동작 버전
  - 초보자도 읽고 수정할 수 있는 구조
  - 이후 실제 데이터/제약/보상 설계가 들어오면 세부 고도화 가능
