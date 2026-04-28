import streamlit as st
import pandas as pd

df_menu = pd.DataFrame({
    '메뉴명':['아메리카노', '카페라떼', '카푸치노', '말차라떼'],
    '가격':[4500, 5000, 5500, 6000]
})

# 데이터프레임 출력-----
# 전체 너비로 늘리기
st.dataframe(df_menu, height=170, width='stretch')

# 원본 크기 유지
st.dataframe(df_menu, height=170, width='content')

# 픽셀 지정 (숫자는 그대로 사용 가능)
st.dataframe(df_menu, height=170, width=400)

st.divider()

# 테이블
st.table(df_menu)