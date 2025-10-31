# Cross-Platform Music Performance Intelligence System
> Spotify-YouTube 통합 분석을 통한 음악 콘텐츠 최적화 전략 수립

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Hypothesis](#hypothesis)
- [Methodology](#methodology)
- [Results](#results)
- [Technical Implementation](#technical-implementation)
- [Business Impact](#business-impact)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Project Overview

Spotify와 YouTube 데이터를 통합 분석하여 히트곡 예측 모델과 크로스 플랫폼 최적화 전략을 제시한 프로젝트
음악 스트리밍 시장에서 플랫폼별 성과 지표가 분산되어 있어 효과적인 콘텐츠 전략 수립이 어려운 문제를 해결하기 위함! 

### Key Features
- 머신러닝 기반 스트림 수 예측 모델 (R² 0.847)
- 히트곡 분류 모델 (Accuracy 92.3%)
- 크로스 플랫폼 성과 상관관계 분석
- 실시간 KPI 대시보드
- MLflow 기반 실험 관리 시스템

### Tech Stack
- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, Random Forest
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Streamlit
- **Database**: SQLite
- **Statistical Analysis**: SciPy

---

## Problem Statement

### Business Problem
1. 신곡 프로모션 시 플랫폼별 예산 배분에 대한 의사결정 근거 부족
2. Spotify 스트림 수와 YouTube 조회수 간 상관관계 불명확
3. 히트곡의 오디오 특성을 사전에 파악하지 못해 A&R 의사결정 어려움

### Research Questions
- Spotify 스트림 수를 사전에 예측할 수 있는가?
- 어떤 오디오 특성이 크로스 플랫폼 성공에 가장 큰 영향을 미치는가?
- 공식 뮤직비디오 제작의 실제 ROI는 얼마인가?

---

## Hypothesis

### H1: 스트림 예측 가능성
Spotify 스트림 수는 오디오 특성(댄서빌리티, 에너지, 발렌스)과 YouTube 참여도의 조합으로 70% 이상 예측 가능하다.
**Rationale**: 선행 연구에서 음악의 감성적 특성이 소비자 선호도와 강한 상관관계를 보임

### H2: 공식 뮤직비디오 효과
공식 뮤직비디오가 있는 트랙은 없는 트랙 대비 평균 스트림 수가 30% 이상 높다.
**Rationale**: 시각적 콘텐츠가 음악 소비 전환율을 높인다는 마케팅 이론

### H3: 크로스 플랫폼 시너지
크로스 플랫폼 상관계수가 0.6 이상일 경우, 통합 프로모션 전략이 단일 플랫폼 전략 대비 효율적이다.
**Rationale**: 멀티채널 마케팅의 시너지 효과

---

## Methodology

### Data Collection & Preprocessing

#### Dataset
- **Source**: Spotify & YouTube integrated dataset
- **Size**: 20,000+ tracks
- **Features**: 14 audio features + engagement metrics

#### Feature Engineering
```python

