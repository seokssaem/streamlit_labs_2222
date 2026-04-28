import streamlit as st

# code
code = '''
import seaborn as sns

iris = sns.load_dataset('iris')
sns.pairplot(data=iris, hue='species', corner=True)
plt.show()
'''

st.code(code, language='python')

# button
def button_write():
    st.write('button activated')

st.button('Reset', type='primary')
st.button('activate', on_click=button_write)


if st.button('Reset', type='primary', key='btn1'):
    st.write("Reset clicked!")

if st.button('Cancel', type='secondary', key='btn2'):
    st.write("Cancel clicked!")

if st.button('Ignore', type='tertiary', key='btn3'):
    st.write("Ignore clicked!")