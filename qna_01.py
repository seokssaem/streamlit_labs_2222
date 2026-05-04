import streamlit as st

st.set_page_config(
    page_title='나의 첫 Steamlit 앱',
    page_icon='😊',
    layout='wide'
)

st.title('첫 Streamlit 앱')
st.write('streamlit 설치 완료!')

name = st.text_input('이름을 입력하세요')

if name:
    st.success(f'{name}님 streamlit 앱 실행 성공입니다!')