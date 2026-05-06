"""
코로나19 한국 대시보드
=====================
Johns Hopkins University의 COVID-19 공개 데이터셋을 활용하여
대한민국의 확진자 / 사망자 / 회복자 현황을 시각화하는 Streamlit 앱입니다.

사용 데이터 (GitHub에서 다운로드):
  - time_series_covid19_confirmed_global.csv  (확진자)
  - time_series_covid19_deaths_global.csv     (사망자)
  - time_series_covid19_recovered_global.csv  (회복자)

실행 방법:
  streamlit run 10_kovid19.py

핵심 학습 포인트:
  1. st.file_uploader  : 파일 업로드 위젯
  2. pd.read_csv       : CSV → DataFrame 변환
  3. DataFrame 가공    : 필터링, 열 삭제, 합산, 병합, diff()
  4. st.tabs           : 탭 UI
  5. st.multiselect    : 다중 선택 위젯
  6. plotly.express    : 인터랙티브 선/막대/파이 그래프
  7. label_map 패턴   : UI 표시 이름과 DataFrame 열 이름 분리
"""

import streamlit as st       # 웹 앱 프레임워크
import pandas as pd           # 데이터 처리
import plotly.express as px   # 인터랙티브 차트


# ──────────────────────────────────────────────────────────────
# 1. 페이지 기본 설정
# ──────────────────────────────────────────────────────────────
# set_page_config()는 반드시 다른 st 명령보다 먼저 호출해야 함
st.set_page_config(
    page_title="코로나19 한국 대시보드",   # 브라우저 탭에 표시될 제목
    layout="wide"                          # "wide" : 전체 화면 너비 사용
                                           # "centered"(기본값) : 중앙 고정 폭
)
st.title("🇰🇷 코로나19 한국 감염자 대시보드")   # 앱 최상단 제목


# ──────────────────────────────────────────────────────────────
# 2. 파일 업로더
# ──────────────────────────────────────────────────────────────
# st.file_uploader() : 파일을 드래그&드롭 또는 클릭으로 올릴 수 있는 위젯
# - type=["csv"] : CSV 파일만 허용 (확장자 제한)
# - 파일이 업로드되지 않은 상태에서는 None 반환
# - 업로드된 파일은 BytesIO 객체로 반환 → pd.read_csv()에 바로 전달 가능
uploaded_confirmed = st.file_uploader("확진자 CSV 업로드", type=["csv"])
uploaded_deaths    = st.file_uploader("사망자 CSV 업로드", type=["csv"])
uploaded_recovered = st.file_uploader("회복자 CSV 업로드", type=["csv"])


# ──────────────────────────────────────────────────────────────
# 3. 세 파일이 모두 업로드됐을 때만 분석 실행
# ──────────────────────────────────────────────────────────────
# 업로드 전 : 파일 객체 = None → 조건 False → 아래 블록 건너뜀
# 업로드 후 : 파일 객체 존재 → 조건 True  → 분석 실행
if uploaded_confirmed and uploaded_deaths and uploaded_recovered:

    # CSV 파일 → DataFrame으로 읽기
    # pd.read_csv()는 파일 경로 대신 업로드 객체(BytesIO)도 바로 처리 가능
    df_confirmed = pd.read_csv(uploaded_confirmed)
    df_deaths    = pd.read_csv(uploaded_deaths)
    df_recovered = pd.read_csv(uploaded_recovered)


    # ──────────────────────────────────────────────────────────
    # 4. 한국 데이터 추출 함수 정의
    # ──────────────────────────────────────────────────────────
    def get_korea_data(df, value_name):
        """
        글로벌 DataFrame에서 대한민국 행만 추출하여
        날짜-값 형태의 깔끔한 DataFrame으로 반환합니다.

        Parameters
        ----------
        df         : 원본 글로벌 DataFrame (confirmed / deaths / recovered)
        value_name : 결과 열 이름 문자열 (예: 'confirmed', 'deaths', 'recovered')

        Returns
        -------
        DataFrame with columns ['date', value_name]

        원본 데이터 구조 예시:
        | Province/State | Country/Region | Lat  | Long  | 1/22/20 | 1/23/20 | ...
        |----------------|----------------|------|-------|---------|---------|----
        | NaN            | Korea, South   | 36.0 | 128.0 | 1       | 1       | ...
        """

        # ① 대한민국 행 필터링
        #    원본 데이터의 국가명 표기: "Korea, South"
        korea_df = df[df["Country/Region"] == "Korea, South"]

        # ② 분석에 불필요한 열 제거
        #    Province/State, Country/Region, Lat, Long 은 날짜-수치 분석에 불필요
        korea_df = korea_df.drop(columns=["Province/State", "Country/Region", "Lat", "Long"])

        # ③ 날짜별 합산
        #    같은 국가라도 Province(지역)가 여러 행으로 나뉠 수 있음 → .sum()으로 합산
        #    .sum() 결과는 Series → .reset_index()로 날짜 문자열을 열로 변환
        korea_series = korea_df.sum().reset_index()

        # ④ 열 이름 지정
        #    reset_index() 후 열 이름이 0, 1 등 숫자로 지정됨 → 의미 있는 이름으로 변경
        korea_series.columns = ['date', value_name]

        # ⑤ 날짜 문자열 → datetime 타입 변환
        #    원본 형식 예시: "1/22/20"
        #    format='%m/%d/%y' 를 명시해야 파싱 오류 없이 변환됨
        #      %m : 월(01~12)   %d : 일(01~31)   %y : 연도 끝 2자리(20 → 2020)
        korea_series['date'] = pd.to_datetime(korea_series['date'], format='%m/%d/%y')

        return korea_series


    # 세 DataFrame을 각각 한국 데이터로 변환
    df_confirmed = get_korea_data(df_confirmed, 'confirmed')
    df_deaths    = get_korea_data(df_deaths,    'deaths')
    df_recovered = get_korea_data(df_recovered, 'recovered')


    # ──────────────────────────────────────────────────────────
    # 5. 세 DataFrame 병합 (merge)
    # ──────────────────────────────────────────────────────────
    # 'date' 열을 기준(on='date')으로 inner join (기본값)
    # 확진자 ← 사망자 ← 회복자 순서로 순차 병합
    # 결과 열 구성: date | confirmed | deaths | recovered
    df_merged = df_confirmed.merge(df_deaths, on='date').merge(df_recovered, on='date')

    # datetime → date 타입으로 변환
    # 이유: datetime 타입은 차트 x축에 "2020-01-22 00:00:00" 처럼 시간이 붙어 표시됨
    #       .dt.date 로 변환하면 "2020-01-22" 형태로 깔끔하게 표시됨
    df_merged['date'] = df_merged['date'].dt.date


    # ──────────────────────────────────────────────────────────
    # 6. 일일 신규 수치 계산
    # ──────────────────────────────────────────────────────────
    # .diff()  : 현재 행 - 이전 행 (전날 대비 증가량)
    #            첫 번째 행은 이전 행이 없어 NaN 발생 → fillna(0)으로 0 처리
    # .astype(int) : diff() 후 float 타입이 됨 → 소수점 제거를 위해 int로 변환
    df_merged['new_confirmed'] = df_merged['confirmed'].diff().fillna(0).astype(int)
    df_merged['new_deaths']    = df_merged['deaths'].diff().fillna(0).astype(int)
    df_merged['new_recovered'] = df_merged['recovered'].diff().fillna(0).astype(int)

    # 최종 df_merged 열 구성:
    # date | confirmed | deaths | recovered | new_confirmed | new_deaths | new_recovered


    # ──────────────────────────────────────────────────────────
    # 7. 탭 UI 구성
    # ──────────────────────────────────────────────────────────
    # st.tabs() : 탭 레이블 리스트를 넘기면 탭 객체 리스트 반환
    # with 블록 안에 작성한 요소가 해당 탭 영역에만 표시됨
    tab1, tab2, tab3 = st.tabs(["📈 감염 추이", "📊 통계 요약", "⚖️ 비율 분석"])


    # ── Tab 1 : 감염 추이 ────────────────────────────────────
    with tab1:

        # ────────────────────────────────────────────────────
        # [핵심 패턴] label_map : UI 표시 이름 ↔ DataFrame 열 이름 분리
        # ────────────────────────────────────────────────────
        # 문제:
        #   multiselect의 options에 'confirmed' 같은 영어 열 이름을 직접 넣으면
        #   화면에 영어가 그대로 노출되어 사용자가 알아보기 어려움
        #
        # 해결:
        #   dict(사전)으로 한글 레이블 → 영어 열 이름 매핑 테이블을 만든다.
        #   { '화면에 보일 한글 이름': '실제 DataFrame 열 이름' }
        #
        #   multiselect에는 dict의 키(한글)를 보여주고,
        #   사용자가 선택한 한글을 dict로 조회해 영어 열 이름으로 변환 후 사용
        # ────────────────────────────────────────────────────

        # ① 누적 추이 (선 그래프) ─────────────────────────────
        st.subheader("📈 누적 추이 그래프")

        # 누적 데이터용 매핑 사전
        label_map = {
            '확진자': 'confirmed',
            '사망자': 'deaths',
            '회복자': 'recovered'
        }

        # st.multiselect : 체크박스 형태의 다중 선택 위젯
        # options : list(label_map.keys()) → ['확진자', '사망자', '회복자'] (한글만 표시)
        # default : 앱 시작 시 미리 선택될 항목
        selected_labels = st.multiselect(
            "표시할 항목을 선택하세요",
            options=list(label_map.keys()),
            default=['확진자', '회복자']
        )

        if selected_labels:  # 하나 이상 선택됐을 때만 그래프 출력
            # 선택된 한글 레이블 리스트 → 영어 열 이름 리스트로 변환
            # 예) ['확진자', '회복자'] → ['confirmed', 'recovered']
            selected_cols = [label_map[label] for label in selected_labels]

            # px.line : Plotly 선 그래프
            # y=selected_cols  : 영어 열 이름으로 DataFrame에서 데이터 조회
            # labels           : 차트 내부(범례, 축 툴팁)에 표시될 이름을 한글로 변환
            #   { '영어 열 이름': '한글 표시 이름' } 형태 필요
            #   → dict comprehension으로 label_map을 뒤집어서 생성
            #   → { 'confirmed': '확진자', 'deaths': '사망자', 'recovered': '회복자' }
            fig = px.line(
                df_merged,
                x="date",
                y=selected_cols,
                markers=True,   # 각 날짜 지점에 점(marker) 표시
                labels={col: kor for kor, col in label_map.items()}
            )
            st.plotly_chart(fig, width='stretch')  # width='stretch' : 컨테이너 전체 너비

        # ② 일일 신규 수치 (막대 그래프) ─────────────────────
        st.subheader("🆕 일일 증가량 그래프")

        # 신규 데이터용 매핑 사전 (구조는 누적용과 동일)
        new_label_map = {
            '신규 확진자': 'new_confirmed',
            '신규 사망자': 'new_deaths',
            '신규 회복자': 'new_recovered'
        }

        selected_new_labels = st.multiselect(
            "표시할 항목 (신규)",
            options=list(new_label_map.keys()),
            default=['신규 확진자']
        )

        if selected_new_labels:
            # 선택된 한글 레이블 → 영어 열 이름으로 변환
            selected_new_cols = [new_label_map[label] for label in selected_new_labels]

            # px.bar : Plotly 막대 그래프
            # labels 파라미터로 범례·축 툴팁을 한글로 표시
            fig_new = px.bar(
                df_merged,
                x="date",
                y=selected_new_cols,
                labels={col: kor for kor, col in new_label_map.items()}
            )
            st.plotly_chart(fig_new, width='stretch')


    # ── Tab 2 : 통계 요약 ────────────────────────────────────
    with tab2:
        st.subheader("📋 일자별 통계 테이블")

        # .tail(10) : 가장 최근 10일 데이터만 슬라이싱
        # st.dataframe : 정렬·검색이 가능한 인터랙티브 테이블로 렌더링
        st.dataframe(df_merged.tail(10), width='stretch')


    # ── Tab 3 : 비율 분석 ────────────────────────────────────
    with tab3:
        st.subheader("⚖️ 최신일 기준 회복률 / 치명률")

        # .iloc[-1] : 인덱스 기준 마지막 행 = 가장 최근 날짜 데이터
        latest    = df_merged.iloc[-1]
        confirmed = latest['confirmed']
        deaths    = latest['deaths']
        recovered = latest['recovered']

        # 비율 계산 (0으로 나누기 방지: confirmed가 0이면 비율도 0으로 처리)
        recovery_rate = (recovered / confirmed) * 100 if confirmed else 0
        fatality_rate = (deaths    / confirmed) * 100 if confirmed else 0

        # st.columns(2) : 화면을 2열로 균등 분할
        col1, col2 = st.columns(2)

        # st.metric : 수치를 강조해서 보여주는 카드형 위젯
        # f"{값:.2f} %" : 소수점 2자리까지 표시
        col1.metric("✅ 회복률", f"{recovery_rate:.2f} %")
        col2.metric("☠️ 치명률", f"{fatality_rate:.2f} %")

        st.subheader("📊 감염자 분포 비율")

        # 파이차트용 데이터 직접 생성
        # active(격리중) = 확진자 - 회복자 - 사망자 (현재 치료/격리 중인 인원)
        pie_df = pd.DataFrame({
            'category': ['회복자', '사망자', '격리중'],
            'count':    [recovered, deaths, confirmed - recovered - deaths]
        })

        # px.pie : Plotly 파이 차트
        # names='category' : 범례에 표시될 항목 이름 열
        # values='count'   : 각 항목의 수치 열
        fig_pie = px.pie(pie_df, names='category', values='count', title='감염자 분포')
        st.plotly_chart(fig_pie, width='stretch')


# ──────────────────────────────────────────────────────────────
# 8. 파일 미업로드 상태 안내 메시지
# ──────────────────────────────────────────────────────────────
else:
    # if 조건이 False일 때 (파일 3개 중 하나라도 업로드 안 된 경우)
    # st.info : 파란색 안내 박스로 사용자에게 다음 행동을 안내
    st.info("3개의 CSV 파일(확진자, 사망자, 회복자)을 모두 업로드 해주세요.")
