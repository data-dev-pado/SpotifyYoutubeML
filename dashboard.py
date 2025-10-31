import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

st.title("Spotify - YouTube 통합 분석 대시보드")

@st.cache_data
def load_data():
    return pd.read_csv('data/processed_spotify_youtube.csv')

@st.cache_data
def load_kpis():
    with open('reports/executive_report.json', 'r') as f:
        report = json.load(f)
    return report['performance_metrics']

@st.cache_data
def load_model_metrics():
    with open('reports/executive_report.json', 'r') as f:
        report = json.load(f)
    return report['model_performance']

df = load_data()
kpis = load_kpis()
model_metrics = load_model_metrics()

st.sidebar.header("필터")
unique_artists = df['Artist'].unique()
selected_artist = st.sidebar.selectbox("아티스트 선택", options=['전체'] + list(unique_artists))

if selected_artist != '전체':
    filtered_df = df[df['Artist'] == selected_artist]
else:
    filtered_df = df

# KPI 보여주기
st.header("핵심 성과 지표 (KPI)")
col1, col2, col3 = st.columns(3)
col1.metric("총 트랙 수", f"{kpis['total_tracks']:,}")
col2.metric("총 아티스트 수", f"{kpis['total_artists']:,}")
col3.metric("총 스트림 수", f"{kpis['total_streams']:,}")

col4, col5, col6 = st.columns(3)
col4.metric("평균 스트림/트랙", f"{kpis['avg_streams_per_track']:.0f}")
col5.metric("크로스 플랫폼 상관관계", f"{kpis['cross_platform_correlation']:.3f}")
col6.metric("평균 참여율", f"{kpis['avg_engagement_rate']*100:.2f} %")

st.markdown("---")

# 스트림 분포 시각화
st.header("스트림 수 분포")
fig, ax = plt.subplots(figsize=(10,4))
sns.histplot(filtered_df['Stream'], bins=50, kde=True, ax=ax)
ax.set_xlabel('Stream 수')
ax.set_ylabel('트랙 수')
ax.set_title('스트림 수 분포')
st.pyplot(fig)

# 예측 모델 성능 요약
st.header("모델 성능 요약")
st.write(f"- 스트림 예측 R²: {model_metrics.get('stream_r2', 0):.3f}")
st.write(f"- 히트 예측 정확도: {model_metrics.get('hit_accuracy', 0)*100:.1f}%")
st.write(f"- YouTube 조회수 예측 R²: {model_metrics.get('youtube_r2', 0):.3f}")

# 아티스트별 평균 스트림 상위 10개 표시
st.header("아티스트별 평균 스트림 (상위 10)")
artist_streams = df.groupby('Artist')['Stream'].mean().sort_values(ascending=False).head(10)
st.bar_chart(artist_streams)

# 상세 데이터 테이블 
st.header("상세 트랙 데이터")
st.dataframe(filtered_df[['Track', 'Artist', 'Stream', 'Views', 'Likes', 'Comments']].reset_index(drop=True))

st.markdown("---")
st.caption("데이터 출처: Spotify_Youtube.csv, 분석 및 모델링: SpotifyYouTubeAnalytics")
