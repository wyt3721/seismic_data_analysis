# import yaml
import streamlit as st
from streamlit.runtime.state import SessionState
# from yaml.loader import SafeLoader
# import streamlit_authenticator as stauth
import pandas as pd
import datetime
# from streamlit_authenticator.utilities.exceptions import (CredentialsError,
#                                                           ForgotError,
#                                                           LoginError,
#                                                           RegisterError,
#                                                           ResetError,
#                                                           UpdateError)


st.set_page_config(page_title='首页',
                   page_icon=":house:",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items={'Report a bug': "mailto:wyt3721@outlook.com",
                               'About': 'This is a demo application'}
                   )


# with open('./.streamlit/config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)

# hashed_passwords = stauth.Hasher(passwords).generate()

# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# authenticator.login()


# if not st.session_state["authentication_status"]:
#     st.warning("请登录 ")


# elif st.session_state["authentication_status"] is None:
#     st.warning('请输入用户名和密码')

# elif st.session_state["authentication_status"] is True:

    # st.success('登录成功')
st.balloons()
# st.sidebar.write(f'你好， *{st.session_state["name"]}*')
st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                     :email: wyt3721@outlook.com
                      Ver0.1a''')
# authenticator.logout("退出", "sidebar")

st.image("./images/qlu.png")
st.title(" :earth_asia:地震大数据分析可视化系统")
# st.audio("./images/qq.wav", format="audio/wav", loop=False, autoplay=False)

st.divider()

# 显示精确时间
time = datetime.datetime.now().astimezone()
time = time.strftime("%Y-%m-%d %H:%M:%S%Z")

st.write("现在是：", time)

st.markdown(
  """
    *:warning:实时地震快报：*
  """
)
st.caption('（根据中国地震台网）')
# st.info("注： :blue[time为UTC时间,北京时间 = UTC + 8:00]")
# usgs 数据源：
# url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv'
# df = pd.read_csv(url)
# st.write(df.style.highlight_max(''))
# 取出经纬度两列，然后传参数给 st.map()
# data2 = df.iloc[:, 1:3]
# st.map(data2, latitude='latitude', longitude='longitude', use_container_width=True)
# st.map(data, latitude='latitude', longitude='longitude', use_container_width=True)

# 中国地震台数据源：
url = 'https://news.ceic.ac.cn/index.html'
# 网页结构简单， 用pandas 可简单爬取网页表格数据:
df = pd.read_html(url)[0]
# 显示表格：
df = df.head(10)
st.dataframe(df)
# 取出经纬度两列数据
data = df.iloc[:, 2:4]
# st.map 不用中文经纬度作为列名
data.rename(columns={'纬度(°)': 'latitude', '经度(°)': 'longitude'}, inplace=True)
# 经纬度两列重命名后传参给 st.map()
st.map(data,  use_container_width=True)







