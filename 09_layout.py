"""
Streamlit 레이아웃 기초 실습
=============================
이 파일은 Streamlit의 핵심 레이아웃 요소를 한 파일에서 비교·실습하기 위한 예제입니다.

학습 포인트:
  1. 사이드바 (st.sidebar)
  2. 이미지 출력 (st.image)
  3. 컬럼 레이아웃 (st.columns)
  4. 탭 레이아웃 (st.tabs)
  5. seaborn + matplotlib 차트를 Streamlit에 삽입하는 방법

필요 파일 (./input/ 폴더 안에 있어야 함):
  - image2.jpg
  - image3.jpg
  - medical_cost.csv

실행 방법:
  streamlit run layout_demo.py
"""

import streamlit as st          # 웹 앱 프레임워크
import pandas as pd             # 데이터 처리
import seaborn as sns           # 통계 시각화 라이브러리
import matplotlib.pyplot as plt # seaborn의 기반 그래프 도구
from PIL import Image           # 이미지 파일 열기 (Pillow 라이브러리)


# ──────────────────────────────────────────────
# 1. 메인 페이지 제목
# ──────────────────────────────────────────────
# st.title : 가장 큰 제목 (h1 수준)
st.title('This is main page')


# ──────────────────────────────────────────────
# 2. 사이드바 (Sidebar)
# ──────────────────────────────────────────────
# with st.sidebar 블록 안에 작성한 요소는
# 화면 왼쪽 사이드바에 표시됨
with st.sidebar:
    st.title('This is sidebar')

    # st.multiselect : 여러 항목을 동시에 선택할 수 있는 드롭다운 위젯
    # label     : 위젯 위에 표시되는 설명 텍스트
    # options   : 선택 가능한 항목 리스트
    # placeholder : 아무것도 선택하지 않았을 때 표시되는 안내 문구
    # 반환값    : 사용자가 선택한 항목들의 리스트 (미선택 시 빈 리스트 [])
    side_option = st.multiselect(
        label='your selection is',
        options=['Car', 'Airplane', 'Train', 'Ship', 'Bicycle'],
        placeholder='select transportation'
    )
    # 선택값 확인 예시 (주석 해제하면 사이드바에 결과 출력)
    # st.write(side_option)


# ──────────────────────────────────────────────
# 3. 이미지 불러오기
# ──────────────────────────────────────────────
# Image.open() : PIL(Pillow) 라이브러리로 이미지 파일을 열어 객체로 저장
# 경로는 이 .py 파일 기준 상대 경로
img2 = Image.open('./input/image2.jpg')
img3 = Image.open('./input/image3.jpg')


# ──────────────────────────────────────────────
# 4. 이미지 세로 나열 (비교용 - 컬럼 미사용)
# ──────────────────────────────────────────────
# st.header : 소제목 (h2 수준)
# st.image  : 이미지 출력
#   width   : 픽셀 단위 너비 (생략하면 원본 크기)
#   caption : 이미지 아래 설명 텍스트
st.header('Lemonade')
st.image(img2, width=300, caption='Image from Unsplash')

st.header('Cocktail')
st.image(img3, width=300, caption='Image from Unsplash')
# → 위 두 이미지는 위아래로 나열됨 (기본 레이아웃)


# ──────────────────────────────────────────────
# 5. 컬럼 레이아웃 (st.columns)
# ──────────────────────────────────────────────
# st.columns(2) : 화면을 동일한 너비의 2열로 분할
# 반환값 : 각 열에 해당하는 컬럼 객체 리스트
# 비율 지정 예시 → st.columns([2, 1])  : 2:1 비율로 분할
col1, col2 = st.columns(2)

# with col1 블록 안의 요소는 왼쪽 열에 표시
with col1:
    st.header('Lemonade')
    st.image(img2, width=300, caption='Image from Unsplash')

# with col2 블록 안의 요소는 오른쪽 열에 표시
with col2:
    st.header('Cocktail')
    st.image(img3, width=300, caption='Image from Unsplash')
# → 두 이미지가 나란히 좌우로 배치됨 (4번과 비교)


# ──────────────────────────────────────────────
# 6. 탭 레이아웃 (st.tabs)
# ──────────────────────────────────────────────
# st.tabs() : 탭 레이블 리스트를 넘기면 탭 객체 리스트 반환
# 탭 수만큼 변수를 받아 각 with 블록에서 사용
tab1, tab2 = st.tabs(['Table', 'Graph'])


# ── 데이터 준비 ───────────────────────────────
# CSV 파일 읽기
df = pd.read_csv('./input/medical_cost.csv')

# .query() : SQL WHERE 절처럼 조건으로 행 필터링
# 'region == "northwest"' → region 열이 northwest 인 행만 남김
df = df.query('region == "northwest"')


# ── Tab 1 : 데이터 테이블 ────────────────────
with tab1:
    # st.table : 정적 테이블 (정렬·스크롤 불가)
    # st.dataframe 과 달리 인터랙션 없이 깔끔하게 출력
    # .head(5) : 상위 5행만 표시
    st.table(df.head(5))


# ── Tab 2 : 산점도 그래프 ────────────────────
with tab2:
    # matplotlib Figure 객체 생성
    # fig : 전체 그림 캔버스 / ax : 그래프가 그려지는 축(Axes)
    fig, ax = plt.subplots()

    # sns.scatterplot : seaborn 산점도
    # data  : 사용할 DataFrame
    # x, y  : 각 축에 매핑할 열 이름
    # ax=ax : matplotlib ax 위에 그리도록 지정 (Streamlit 연동 필수)
    sns.scatterplot(data=df, x='bmi', y='charges', ax=ax)

    # st.pyplot(fig) : matplotlib/seaborn 그래프를 Streamlit에 표시
    # fig 를 명시적으로 전달하지 않으면 경고 발생하므로 반드시 넘길 것
    st.pyplot(fig)