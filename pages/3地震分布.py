import streamlit as st
import numpy as np
import pandas as pd
# import yaml
# from yaml.loader import SafeLoader
# import streamlit_authenticator as stauth
import time
# from obspy.core import UTCDateTime
from datetime import datetime

# import plotly.graph_objects as go
# import plotly.express as px
# import pydeck as pdk

# with open('./.streamlit/config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# if not st.session_state["authentication_status"]:
#     st.switch_page('app.py')
# st.sidebar.success('''欢迎使用本系统 :smile: :joy:
#                        :email: wyt3721@outlook.com
#                         Ver0.1a''')


@st.cache_data
def read_data():
    data = pd.read_csv("./data/2000-2023.csv")
    return data



# if st.session_state["authentication_status"]:
st.title("全球地震分布图")
with st.sidebar:
    # st.title("地震年份")
    opinions = st.sidebar.selectbox("请选择年份：", (
                                        '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
                                        '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
                                        '2018', '2019', '2020', '2021', '2022', '2023'))
df = read_data()
df['time'] = pd.to_datetime(df['time'])
df['year'] = pd.DatetimeIndex(df['time']).year.astype('str')

df_y = df.loc[df['year'] == opinions]

st.map(df_y, latitude='latitude', longitude='longitude')
# 卫星云图 向日葵八号
now = datetime.now().strftime('%Y%m%d')
st.write('实时参考卫星云图')
# video_url = 'https://ncthmwrwbtst.cr.chiba-u.ac.jp/movie/720/' + now + '_pifd.mp4'
video_url = 'https://himawari8.nict.go.jp/movie/720/' + now + '_pifd.mp4'
video_url_coastline = 'https://himawari8.nict.go.jp/movie/720/coastline/' + now + '_pifd.mp4'

st.video(video_url, autoplay=False)
st.video(video_url_coastline, autoplay=False)
