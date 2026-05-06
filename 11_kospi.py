"""
KOSPI 주식 차트 대시보드
========================
FinanceDataReader 라이브러리를 활용하여
KOSPI 시장의 종목별 주가 데이터를 시각화하는 Streamlit 앱입니다.

설치 필요 라이브러리:
  uv add finance-datareader plotly

실행 방법:
  streamlit run 11_kospi.py

핵심 학습 포인트:
  1. FinanceDataReader  : 주식 데이터 수집 (한국/해외 시장)
  2. fdr.StockListing() : 시장 전체 종목 리스트 조회
  3. fdr.DataReader()   : 개별 종목 가격 데이터 조회
  4. go.Bar()           : Plotly 수평 막대그래프 (시가총액 TOP10)
  5. go.Candlestick()   : Plotly 캔들스틱 차트 (시가/고가/저가/종가)
  6. st.sidebar         : 사이드바 UI 구성
  7. st.metric()        : 현재가/변동폭 카드 표시
  8. st.tabs()          : 탭 UI (라인 차트 / 캔들스틱 차트)
  9. unicodedata        : 한글 종목명 정규화 (띄어쓰기/특수문자 처리)
"""

import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd
import plotly.graph_objects as go
import unicodedata


# ──────────────────────────────────────────────────────────────
# 1. 페이지 기본 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='주식 차트 대시보드',
    page_icon='📈'
)
st.title('📈 KOSPI 주식 차트 대시보드')


# ──────────────────────────────────────────────────────────────
# 2. 한글 종목명 정규화 함수
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# 3. KOSPI 전체 종목 리스트 로드
# ──────────────────────────────────────────────────────────────
# fdr.StockListing(market) : 해당 시장의 전체 종목 DataFrame 반환
# 주요 컬럼: Code(종목코드), Name(종목명), Marcap(시가총액), Sector(업종) 등
market = 'KOSPI'
df_market = fdr.StockListing(market)

# 종목명 정규화 적용 (비교 오류 방지)
df_market['Name'] = df_market['Name'].apply(normalize_str)

# multiselect 위젯에 사용할 종목명 리스트 추출
stocks = df_market['Name'].tolist()


# ──────────────────────────────────────────────────────────────
# 4. 시가총액 TOP 10 수평 막대그래프
# ──────────────────────────────────────────────────────────────
# .nlargest(10, 'Marcap') : 시가총액 기준 상위 10개 행 추출
# .iloc[::-1]             : 행 순서를 뒤집음
#   → 수평 막대그래프에서 위쪽에 1위가 오도록 역순 정렬
top10 = df_market.nlargest(10, 'Marcap').iloc[::-1]

fig_top10 = go.Figure(go.Bar(
    x=top10['Marcap'] / 1e12,      # 시가총액 단위 변환: 원 → 조 (1조 = 1e12)
    y=top10['Name'],               # y축: 종목명
    orientation='h',               # 'h': 수평(horizontal) 막대그래프
                                   # 'v': 수직(vertical, 기본값)
    text=top10['Marcap'] / 1e12,  # 막대 위에 표시할 텍스트 (조 단위 숫자)
    texttemplate='%{text:.1f}조'   # 텍스트 포맷: 소수점 1자리 + '조' 단위 표시
))

fig_top10.update_layout(
    title=f'{market} 시가총액 TOP10',
    xaxis_title='시가총액 (조)',
    yaxis_title='종목명',
    bargap=0.15   # 막대 사이 간격 (0~1 사이 값, 클수록 간격 넓어짐)
)

st.plotly_chart(fig_top10, width='stretch')


# ──────────────────────────────────────────────────────────────
# 5. 사이드바: 종목 선택
# ──────────────────────────────────────────────────────────────
# st.sidebar.multiselect : 사이드바 영역에 다중 선택 위젯 배치
# max_selections=10      : 최대 10개까지만 선택 가능 (성능 고려)
selected_stocks = st.sidebar.multiselect(
    '종목을 선택하세요 (최대 10개)',
    stocks,
    max_selections=10
)

# 선택된 종목명도 동일하게 정규화 (df_market의 Name과 비교하기 위해)
selected_stocks = [normalize_str(s) for s in selected_stocks]


# ──────────────────────────────────────────────────────────────
# 6. 선택된 종목명 → 종목 코드로 변환
# ──────────────────────────────────────────────────────────────
# FinanceDataReader의 DataReader()는 종목명이 아닌 종목 코드를 입력받음
# 예) '삼성전자' → '005930'
#
# df_market.loc[조건, '열이름'].values
#   : 조건에 맞는 행의 특정 열 값을 numpy 배열로 반환
#   : .values[0] 으로 첫 번째 값(코드 문자열)을 꺼냄
codes = []
for name in selected_stocks:
    matched = df_market.loc[df_market['Name'] == name, 'Code'].values
    if len(matched) > 0:
        codes.append(matched[0])

# 선택된 종목이 없거나 코드 변환에 실패한 경우 경고 후 실행 중단
# st.stop() : 이 줄 이후의 코드를 실행하지 않음 (early return과 유사)
if not codes:
    st.warning("종목 코드를 찾을 수 없습니다. 종목을 다시 선택해주세요.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# 7. 사이드바: 날짜 범위 선택
# ──────────────────────────────────────────────────────────────
# st.sidebar.date_input : 날짜 선택 위젯 (캘린더 UI)
# 두 번째 인자: 기본값 (앱 시작 시 미리 선택되는 날짜)
start_date = st.sidebar.date_input('시작 날짜', datetime.date(2022, 1, 1))
end_date   = st.sidebar.date_input('종료 날짜', datetime.datetime.now().date())


# ──────────────────────────────────────────────────────────────
# 8. 주식 데이터 로드 함수
# ──────────────────────────────────────────────────────────────
def get_stock_data(code, start, end):
    """
    FinanceDataReader로 개별 종목의 주가 데이터를 조회합니다.

    Parameters
    ----------
    code  : 종목 코드 문자열 (예: '005930')
    start : 조회 시작일 문자열 (예: '2022-01-01')
    end   : 조회 종료일 문자열 (예: '2026-05-06')

    Returns
    -------
    DataFrame (columns: Open, High, Low, Close, Volume 등)
    조회 실패 또는 데이터 없을 시 None 반환

    반환 DataFrame 주요 컬럼:
      Open   : 시가 (당일 첫 거래 가격)
      High   : 고가 (당일 최고 가격)
      Low    : 저가 (당일 최저 가격)
      Close  : 종가 (당일 마지막 거래 가격)
      Volume : 거래량
    """
    try:
        df = fdr.DataReader(code, start, end)
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"{code} 데이터 로드 실패: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# 9. 현재가 / 전일 대비 변동폭 카드 표시
# ──────────────────────────────────────────────────────────────
# st.metric() : 수치를 강조해서 보여주는 카드형 위젯
#   label : 카드 상단에 표시되는 이름
#   value : 현재 수치 (크게 표시)
#   delta : 변동폭 (양수=초록↑, 음수=빨강↓ 자동 색상 적용)
#
# f"{숫자:,}" : 천 단위 구분자(,) 추가 → 1601000 → 1,601,000
for i, code in enumerate(codes):
    df = get_stock_data(
        code,
        start_date.strftime('%Y-%m-%d'),  # date → 문자열 변환 (fdr 입력 형식)
        end_date.strftime('%Y-%m-%d')
    )
    if df is not None and len(df) >= 2:
        current = df['Close'].iloc[-1]    # 가장 최근 종가 (.iloc[-1]: 마지막 행)
        prev    = df['Close'].iloc[-2]    # 전일 종가    (.iloc[-2]: 마지막-1 행)
        delta   = current - prev          # 전일 대비 변동폭

        st.metric(
            label=selected_stocks[i],
            value=f"{current:,}원",
            delta=f"{delta:,}원"
        )
    else:
        st.warning(f"{selected_stocks[i]} 데이터가 충분하지 않습니다.")


# ──────────────────────────────────────────────────────────────
# 10. 탭 UI: 라인 차트 / 캔들스틱 차트
# ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(['라인 차트', '캔들스틱 차트'])


# ── Tab 1 : 라인 차트 ────────────────────────────────────────
with tab1:

    if len(codes) == 1:
        # ① 종목 1개 선택 시: 단순 라인 차트
        df = get_stock_data(
            codes[0],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if df is not None:
            # st.line_chart : Streamlit 기본 내장 라인 차트 (Plotly보다 간단)
            st.line_chart(df['Close'])
        else:
            st.warning("데이터를 불러올 수 없습니다.")

    else:
        # ② 종목 여러 개 선택 시: 종가 데이터를 날짜 기준으로 병합 후 라인 차트
        #
        # 병합 전략:
        #   각 종목의 Close 열만 추출 → 열 이름을 종목코드로 변경 → 리스트에 저장
        #   pd.concat(axis=1): 수평 방향으로 이어 붙임 (날짜 인덱스 기준 자동 정렬)
        #
        # 예시 결과 (merged_df):
        #             005930   000660
        # 2022-01-03  78000   130000
        # 2022-01-04  77500   128000
        dfs = []
        for code in codes:
            df = get_stock_data(
                code,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if df is not None:
                df_temp = df[['Close']].rename(columns={'Close': code})
                dfs.append(df_temp)

        if dfs:
            merged_df = pd.concat(dfs, axis=1)    # 수평 병합
            merged_df.columns = selected_stocks   # 열 이름을 종목명(한글)으로 변경
            st.line_chart(merged_df)
        else:
            st.warning("선택한 종목의 데이터를 불러올 수 없습니다.")


# ── Tab 2 : 캔들스틱 차트 ────────────────────────────────────
with tab2:
    # 캔들스틱 차트: 하루의 주가 흐름을 시가/고가/저가/종가로 표현
    #
    # 캔들 구조:
    #   ─ 고가 (High)
    #   █ 종가 > 시가 → 양봉 (초록/파랑): 주가 상승
    #   █ 종가 < 시가 → 음봉 (빨강): 주가 하락
    #   ─ 저가 (Low)
    for i, code in enumerate(codes):
        df = get_stock_data(
            code,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if df is not None:
            fig_candle = go.Figure(data=[go.Candlestick(
                x=df.index,       # x축: 날짜 (DataFrame 인덱스)
                open=df['Open'],  # 시가
                high=df['High'],  # 고가
                low=df['Low'],    # 저가
                close=df['Close'] # 종가
            )])

            fig_candle.update_layout(
                title=f'{selected_stocks[i]} 캔들스틱 차트',
                xaxis_title='날짜',
                yaxis_title='가격 (원)',
                xaxis_rangeslider_visible=False  # 하단 범위 슬라이더 숨김 (화면 절약)
            )
            st.plotly_chart(fig_candle, width='stretch')
        else:
            st.warning(f"{selected_stocks[i]} 캔들스틱 차트를 불러올 수 없습니다.")
