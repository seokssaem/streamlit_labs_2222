import streamlit as st
import FinanceDataReader as fdr
import datetime
import pandas as pd
import plotly.graph_objects as go
import unicodedata

# Streamlit 페이지 설정: 타이틀과 아이콘 지정
st.set_page_config(page_title='주식 차트 대시보드', page_icon='📈')
st.title('📈 KOSPI 주식 차트 대시보드')

# 문자열 정규화 함수: 한글 종목명 간 띄어쓰기/특수문자 문제 방지용
def normalize_str(s):
    return unicodedata.normalize('NFKC', s).strip()

# 1) KOSPI 시장 전체 종목 정보 가져오기
market = 'KOSPI'
df_market = fdr.StockListing(market)
df_market['Name'] = df_market['Name'].apply(normalize_str)  # 종목명 정규화
stocks = df_market['Name'].tolist()  # 종목명 리스트 추출

# 2) 시가총액 상위 10개 종목 막대그래프 생성
top10 = df_market.nlargest(10, 'Marcap').iloc[::-1]  # 상위 10개 시가총액 역순(막대그래프 수평 정렬을 위해)
fig = go.Figure(go.Bar(
    x=top10['Marcap'] / 1e12,       # 시가총액을 '조' 단위로 변환
    y=top10['Name'],                # 종목명 (y축)
    orientation='h',                # 수평 막대그래프
    text=top10['Marcap'] / 1e12,   # 그래프에 표시할 텍스트 (조 단위)
    texttemplate='%{text:.1f}조'    # 텍스트 포맷 지정 (소수점 1자리, 조 단위)
))
fig.update_layout(
    title=f'{market} 시가총액 TOP10',
    xaxis_title='시가총액 (조)',
    yaxis_title='종목명',
    bargap=0.15  # 막대 간 간격 조정
)
st.plotly_chart(fig)  # Streamlit 화면에 그래프 출력

# 3) 사이드바에서 종목 선택 (최대 10개)
selected_stocks = st.sidebar.multiselect(
    '종목을 선택하세요 (최대 10개)',  # 안내 문구
    stocks,                        # 선택할 수 있는 옵션 목록
    max_selections=10              # 최대 선택 가능 개수 제한
)
selected_stocks = [normalize_str(s) for s in selected_stocks]  # 선택 종목명도 정규화 처리

# 4) 선택된 종목명을 코드로 변환 (FinanceDataReader는 종목 코드 필요)
codes = []
for name in selected_stocks:
    matched = df_market.loc[df_market['Name'] == name, 'Code'].values  # 이름에 맞는 코드 검색
    st.sidebar.write(f"선택: {name} -> 코드: {matched}")  # 디버깅용 출력
    if len(matched) > 0:
        codes.append(matched[0])  # 코드 리스트에 추가

# 선택한 종목 코드가 없으면 경고 후 실행 중지
if not codes:
    st.warning("종목 코드를 찾을 수 없습니다. 종목을 다시 선택해주세요.")
    st.stop()

# 5) 날짜 입력: 시작일, 종료일 선택 (기본값 지정)
start_date = st.sidebar.date_input('시작 날짜', datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input('종료 날짜', datetime.datetime.now().date())

# 6) 주식 데이터 불러오는 함수 (예외 처리 포함)
def get_stock_data(code, start, end):
    try:
        df = fdr.DataReader(code, start, end)  # FinanceDataReader로 데이터 조회
        if df.empty:
            return None  # 데이터가 없으면 None 반환
        return df
    except Exception as e:
        st.error(f"{code} 데이터 로드 실패: {e}")  # 오류 메시지 출력
        return None

# 7) 선택한 종목별 현재가와 변동폭(전일 대비) 표시
for i, code in enumerate(codes):
    df = get_stock_data(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is not None and len(df) >= 2:
        current = df['Close'].iloc[-1]       # 가장 최신 종가
        prev = df['Close'].iloc[-2]          # 그 전일 종가
        delta = current - prev                # 변동폭 계산
        st.metric(label=selected_stocks[i], value=f"{current:,}원", delta=f"{delta:,}원")  # 화면에 숫자 표시
    else:
        st.warning(f"{selected_stocks[i]} 데이터가 충분하지 않습니다.")  # 데이터 부족 안내

# 8) 그래프 탭: 라인 차트와 캔들스틱 차트 선택 가능
tab1, tab2 = st.tabs(['라인 차트', '캔들스틱 차트'])

with tab1:
    if len(codes) == 1:
        # 종목 하나 선택 시 단일 라인 차트 출력
        df = get_stock_data(codes[0], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df is not None:
            st.line_chart(df['Close'])  # 종가 라인 차트
        else:
            st.warning("데이터를 불러올 수 없습니다.")
    else:
        # 여러 종목 선택 시 종가 데이터 병합 후 라인 차트 출력
        dfs = []
        for code in codes:
            df = get_stock_data(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df is not None:
                # 컬럼명을 종목 코드로 변경해 리스트에 저장
                df_temp = df[['Close']].rename(columns={'Close': code})
                dfs.append(df_temp)
        if dfs:
            merged_df = pd.concat(dfs, axis=1)   # 수평 방향으로 병합
            merged_df.columns = selected_stocks  # 컬럼명에 종목명으로 변경
            st.line_chart(merged_df)              # 라인 차트 출력
        else:
            st.warning("선택한 종목의 데이터를 불러올 수 없습니다.")

with tab2:
    # 캔들스틱 차트: 각 종목별 시가, 고가, 저가, 종가 표시
    for i, code in enumerate(codes):
        df = get_stock_data(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df is not None:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
            )])
            fig.update_layout(
                title=f'{selected_stocks[i]} 캔들스틱 차트',
                xaxis_title='날짜',
                yaxis_title='가격(원)'
            )
            st.plotly_chart(fig)
        else:
            st.warning(f"{selected_stocks[i]} 캔들스틱 차트를 불러올 수 없습니다.")
