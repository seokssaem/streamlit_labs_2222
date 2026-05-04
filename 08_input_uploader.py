
import streamlit as st

# 1. 텍스트 입력 창: 좋아하는 포켓몬 이름 받기
string1 = st.text_input(
    '좋아하는 포켓몬은??',                           # 입력창 라벨
    placeholder='당신이 가장 좋아하는 포켓몬 이름을 적어주세요!',  # 입력창 안내 문구
    max_chars=32                                  # 최대 입력 글자 수 제한
)

# 입력값이 있으면 화면에 출력
if string1:
    st.text(f'Your answer is {string1}')

# 2. 비밀번호 입력 창: 싫어하는 음식 받기 (입력 내용 숨긴다)
string2 = st.text_input(
    '싫어하는 음식은??',                            # 입력창 라벨
    placeholder='당신이 가장 싫어하는 음식을 하나 적어주세요!',  # 안내 문구
    max_chars=32,                                 # 최대 글자 수 제한
    type='password'                              # 입력값을 *로 가린다
)

# 입력값이 있으면 화면에 출력
if string2:
    st.text(f'Your answer is {string2}')

import pandas as pd

# 3. 파일 업로더: CSV 파일만 업로드 가능
file = st.file_uploader(
    'Choose a file',     # 업로드창 라벨
    type='csv',          # 확장자 제한 (csv만)
    accept_multiple_files=False  # 한 번에 하나의 파일만 업로드 가능
)


# 파일이 업로드 되면 판다스로 읽어 데이터프레임 생성 후 화면에 표 형태로 출력
if file is not None:
    df = pd.read_csv(file)
    st.write(df)