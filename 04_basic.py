import streamlit as st
from PIL import Image

image = Image.open('./input/egg.png')

st.image(image, caption='egg_image')
st.image(image, caption='width->100', width=100)
st.image(image, caption='width->200', width=200)

# 화면 꽉 채우기
st.image(image, caption='전체 너비', width='stretch')

# 이미지 원본 크기
st.image(image, caption='원본 크기', width='content')

# 작은 이미지로 테스트
small_image = image.resize((200, 200))  # 강제로 작게

st.image(small_image, caption='stretch - 전체너비', width='stretch')
st.image(small_image, caption='content - 원본크기(200px)', width='content')