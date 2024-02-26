import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(page_title="Лаборатория цифровых компетенций", page_icon="👾", layout="wide")

from pages_list.intro import intro_page
from pages_list.clustering import clustering
from pages_list.visual import visual


with st.sidebar:
    redirect_url = "http://83.143.66.61:27369/"
    logo_image = 'https://raw.githubusercontent.com/Chetoff1228/images/main/logo.png'
    #st.markdown(f'<a href="{redirect_url}"><img src="{logo_image}" alt="Foo" width="100" height="100"/></a>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:grey;'><a href='{redirect_url}'><img src='{logo_image}' alt='Foo' width='100' height='100'/></a></p>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu('', ["1. Что такое машинное обучение?🤔", "2. Введение в кластеризацию🤓", "3. Визуализация🤖", "4. Проверка знаний 🎓", "5. Итоги 📜"])

if selected == "1. Что такое машинное обучение?🤔":
    intro_page('test')
elif selected == "2. Введение в кластеризацию🤓":
    clustering()
elif selected == "3. Визуализация🤖":
    visual()
# elif selected == "4. Проверка знаний 🎓":
#     missuse_page()
# elif selected == "5. Итоги 📜":
#     check_page()
