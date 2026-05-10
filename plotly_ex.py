import plotly.express as px
import plotly.graph_objects as go
import FinanceDataReader as fdr
import unicodedata
import pandas as pd
import streamlit as st

# 0. 데이터
market = 'KOSPI'
df_market = fdr.StockListing(market)

def normalize_str(s):
    """
    한글 문자열을 NFKC 방식으로 정규화합니다.

    필요한 이유:
      FinanceDataReader에서 가져온 종목명과
      multiselect에서 선택한 종목명의 인코딩이
      미묘하게 달라 == 비교 시 False가 되는 경우가 있음.
      normalize()로 통일하면 이 문제를 방지할 수 있음.

    NFKC란?
      유니코드 정규화 방식 중 하나.
      호환 분해(K) + 정준 결합(C) 방식으로
      같은 의미의 문자를 동일한 형태로 통일함.
      예) '＋'(전각 플러스) → '+'(반각 플러스)
          '㈜' → '(주)'
    """
    return unicodedata.normalize('NFKC', s).strip()


# 종목명 정규화 적용 (비교 오류 방지)
df_market['Name'] = df_market['Name'].apply(normalize_str)

# multiselect 위젯에 사용할 종목명 리스트 추출
stocks = df_market['Name'].tolist()

top10 = df_market.nlargest(10, 'Marcap').iloc[::-1]

fig_top10 = go.Figure(go.Bar(
    x=top10['Marcap'] / 1e12,      # 시가총액 단위 변환: 원 → 조 (1조 = 1e12)
    y=top10['Name'],               # y축: 종목명
    orientation='h',               # 'h': 수평(horizontal) 막대그래프
                                   # 'v': 수직(vertical, 기본값)
    text=top10['Marcap'] / 1e12,  # 막대 위에 표시할 텍스트 (조 단위 숫자)
    texttemplate='%{text:.1f}조'   # 텍스트 포맷: 소수점 1자리 + '조' 단위 표시
))

df = pd.read_csv('./input/time_series_covid_19_confirmed.csv')
pie_df = df.copy()

# 1. Plotly Express: 빠른 차트
fig_line = px.line(df, x="date", y=["confirmed", "recovered"], markers=True)
fig_bar = px.bar(df, x="date", y="new_confirmed")
fig_pie = px.pie(pie_df, names="category", values="count")

# 2. Graph Objects: 세밀한 차트
fig_top10 = go.Figure(go.Bar(
    x=top10["Marcap"] / 1e12,
    y=top10["Name"],
    orientation="h",
    text=top10["Marcap"] / 1e12,
    texttemplate="%{text:.1f}조"
))

fig_candle = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)])

# 3. Streamlit 출력
st.plotly_chart(fig_line, width="stretch")