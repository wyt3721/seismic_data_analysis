import streamlit as st
import numpy as np
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
# import plotly.graph_objects as go
# import plotly.express as px
# import pydeck as pdk

with open('./.streamlit/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

if not st.session_state["authentication_status"]:
    st.warning("请登录 :point_down:")


@st.cache_data
def earth_chart():
    data2 = pd.read_csv("./data/2000-2023.csv")
    data2 = data2.iloc[:, 0:3]
    st.write("2000-2023全球地震震中分布图")
    st.write(pd.data2.head())
    # st.map(data2, latitude='latitude', longitude='longitude')


if st.session_state["authentication_status"]:
    earth_chart()


