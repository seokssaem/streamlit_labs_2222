"""
코로나19 한국 대시보드
=====================
Johns Hopkins University의 COVID-19 공개 데이터셋을 활용하여
대한민국의 확진자 / 사망자 / 회복자 현황을 시각화하는 Streamlit 앱입니다.

사용 데이터:
  - time_series_covid19_confirmed_global.csv  (확진자)
  - time_series_covid19_deaths_global.csv     (사망자)
  - time_series_covid19_recovered_global.csv  (회복자)

실행 방법:
  streamlit run 10_kovid19.py
"""

import streamlit as st   # 웹 앱 프레임워크
import pandas as pd      # 데이터 처리
import plotly.express as px  # 인터랙티브 차트

# ──────────────────────────────────────────────
# 1. 페이지 기본 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="코로나19 한국 대시보드",  # 브라우저 탭 제목
    layout="wide"                         # 화면 전체 너비 사용 (기본값은 "centered")
)
st.title("🇰🇷 코로나19 한국 감염자 대시보드")  # 앱 최상단 제목


# ──────────────────────────────────────────────
# 2. 파일 업로더 (사이드바 없이 메인 영역에 배치)
# ──────────────────────────────────────────────
# st.file_uploader : 사용자가 파일을 드래그&드롭 또는 클릭으로 올릴 수 있는 위젯
# type=["csv"] 로 CSV 파일만 허용
uploaded_confirmed = st.file_uploader("확진자 CSV 업로드", type=["csv"])
uploaded_deaths    = st.file_uploader("사망자 CSV 업로드", type=["csv"])
uploaded_recovered = st.file_uploader("회복자 CSV 업로드", type=["csv"])


# ──────────────────────────────────────────────
# 3. 세 파일이 모두 업로드됐을 때만 분석 실행
# ──────────────────────────────────────────────
# 업로드 전에는 파일 객체가 None → 조건 False → 아래 블록 건너뜀
if uploaded_confirmed and uploaded_deaths and uploaded_recovered:

    # CSV → DataFrame 변환
    # pd.read_csv()는 파일 경로 대신 업로드 객체(BytesIO)도 바로 읽을 수 있음
    df_confirmed = pd.read_csv(uploaded_confirmed)
    df_deaths    = pd.read_csv(uploaded_deaths)
    df_recovered = pd.read_csv(uploaded_recovered)


    # ──────────────────────────────────────────
    # 4. 한국 데이터 추출 함수
    # ──────────────────────────────────────────
    def get_korea_data(df, value_name):
        """
        글로벌 DataFrame에서 대한민국 행만 추출하여
        날짜-값 형태의 깔끔한 DataFrame으로 반환합니다.

        Parameters
        ----------
        df         : 원본 글로벌 DataFrame
        value_name : 결과 열 이름 (예: '확진자', '사망자', '회복자')

        Returns
        -------
        DataFrame with columns ['날짜', value_name]
        """

        # ① 대한민국 행 필터링
        #    원본 데이터에서 국가명은 "Korea, South" 로 표기됨
        korea_df = df[df["Country/Region"] == "Korea, South"]

        # ② 분석에 불필요한 열 제거
        #    (Province/State, Country/Region, Lat, Long 은 날짜-수치 분석에 필요 없음)
        korea_df = korea_df.drop(columns=["Province/State", "Country/Region", "Lat", "Long"])

        # ③ 날짜별 합산 (같은 국가에 Province가 여러 행일 경우 대비)
        #    .sum() 이후 .reset_index() 로 인덱스(날짜 문자열)를 열로 변환
        korea_series = korea_df.sum().reset_index()

        # ④ 열 이름 한글로 변경
        korea_series.columns = ['날짜', value_name]

        # ⑤ 날짜 문자열 → datetime 변환
        #    원본 형식 예시: "1/22/20"  →  format='%m/%d/%y' 로 명시해야 오류 없음
        korea_series['날짜'] = pd.to_datetime(korea_series['날짜'], format='%m/%d/%y')

        return korea_series


    # 세 DataFrame 각각 한국 데이터로 변환
    df_confirmed = get_korea_data(df_confirmed, '확진자')
    df_deaths    = get_korea_data(df_deaths,    '사망자')
    df_recovered = get_korea_data(df_recovered, '회복자')


    # ──────────────────────────────────────────
    # 5. 세 DataFrame 병합
    # ──────────────────────────────────────────
    # '날짜' 열을 기준(on='날짜')으로 inner join (기본값)
    # 확진자 ← 사망자 ← 회복자 순서로 순차 병합
    df_merged = df_confirmed.merge(df_deaths, on='날짜').merge(df_recovered, on='날짜')

    # datetime → date 로 변환 (차트 x축에 시간 "00:00:00" 표시 제거)
    df_merged['날짜'] = df_merged['날짜'].dt.date


    # ──────────────────────────────────────────
    # 6. 일일 신규 수치 계산
    # ──────────────────────────────────────────
    # .diff() : 전날 대비 차이 (첫 행은 NaN → fillna(0) 으로 0 처리)
    # .astype(int) : float → int 변환 (소수점 제거)
    df_merged['신규 확진자'] = df_merged['확진자'].diff().fillna(0).astype(int)
    df_merged['신규 사망자'] = df_merged['사망자'].diff().fillna(0).astype(int)
    df_merged['신규 회복자'] = df_merged['회복자'].diff().fillna(0).astype(int)


    # ──────────────────────────────────────────
    # 7. 탭 UI 구성
    # ──────────────────────────────────────────
    # st.tabs() : 탭 레이블 리스트를 넘기면 탭 객체 리스트 반환
    # with 블록 안에 작성한 요소가 해당 탭에 표시됨 
    tab1, tab2, tab3 = st.tabs(["📈 감염 추이", "📊 통계 요약", "⚖️ 비율 분석"])


    # ── Tab 1 : 감염 추이 ──────────────────────
    with tab1:

        # ① 누적 추이 (선 그래프)
        st.subheader("📈 누적 추이 그래프")

        # st.multiselect : 체크박스 형태의 다중 선택 위젯
        # default 로 초기 선택값 지정
        selected = st.multiselect(
            "표시할 항목을 선택하세요",
            ['확진자', '사망자', '회복자'],
            default=['확진자', '회복자']
        )

        if selected:  # 하나 이상 선택됐을 때만 그래프 출력
            # px.line : Plotly 선 그래프
            # y 에 리스트를 넘기면 여러 계열을 한 번에 그림
            # markers=True : 각 날짜 지점에 점 표시
            fig = px.line(df_merged, x="날짜", y=selected, markers=True)

            # width='stretch' : 컨테이너 전체 너비로 차트 확장
            # (구버전의 use_container_width=True 와 동일, 2025-12-31 이후 구버전 제거 예정)
            st.plotly_chart(fig, width='stretch')

        # ② 일일 신규 수치 (막대 그래프)
        st.subheader("🆕 일일 증가량 그래프")

        selected_new = st.multiselect(
            "표시할 항목 (신규)",
            ['신규 확진자', '신규 사망자', '신규 회복자'],
            default=['신규 확진자']
        )

        if selected_new:
            # px.bar : Plotly 막대 그래프
            fig_new = px.bar(df_merged, x="날짜", y=selected_new)
            st.plotly_chart(fig_new, width='stretch')


    # ── Tab 2 : 통계 요약 ─────────────────────
    with tab2:
        st.subheader("📋 일자별 통계 테이블")

        # .tail(10) : 최근 10일 데이터만 표시
        # st.dataframe : 정렬·검색 가능한 인터랙티브 테이블
        st.dataframe(df_merged.tail(10), width='stretch')


    # ── Tab 3 : 비율 분석 ─────────────────────
    with tab3:
        st.subheader("⚖️ 최신일 기준 회복률 / 치명률")

        # .iloc[-1] : 마지막 행 (가장 최근 날짜)
        latest = df_merged.iloc[-1]
        확진자 = latest['확진자']
        사망자 = latest['사망자']
        회복자 = latest['회복자']

        # 0으로 나누기 방지 (확진자가 0이면 비율도 0)
        회복률 = (회복자 / 확진자) * 100 if 확진자 else 0
        치명률 = (사망자 / 확진자) * 100 if 확진자 else 0

        # st.columns(2) : 화면을 2열로 분할
        col1, col2 = st.columns(2)

        # st.metric : 수치를 강조해서 보여주는 카드형 위젯
        col1.metric("✅ 회복률", f"{회복률:.2f} %")
        col2.metric("☠️ 치명률", f"{치명률:.2f} %")

        st.subheader("📊 감염자 분포 비율")

        # 파이차트용 데이터 직접 생성
        # 격리중 = 확진자 - 회복자 - 사망자 (현재 치료/격리 중인 인원)
        pie_df = pd.DataFrame({
            '구분': ['회복자', '사망자', '격리중'],
            '인원수': [회복자, 사망자, 확진자 - 회복자 - 사망자]
        })

        # px.pie : Plotly 파이(도넛) 차트
        fig_pie = px.pie(pie_df, names='구분', values='인원수', title='감염자 분포')
        st.plotly_chart(fig_pie, width='stretch')

# ──────────────────────────────────────────────
# 8. 파일 미업로드 상태 안내 메시지
# ──────────────────────────────────────────────
else:
    # st.info : 파란색 안내 박스
    st.info("3개의 CSV 파일(확진자, 사망자, 회복자)을 모두 업로드 해주세요.") 