# import yaml
import streamlit as st
from streamlit.runtime.state import SessionState
# from yaml.loader import SafeLoader
# import streamlit_authenticator as stauth
import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
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
st.audio("./images/qq.mp3", format="audio/wav", loop=False, autoplay=True)

st.divider()

# 显示精确时间
time = datetime.datetime.now().astimezone()
time = time.strftime("%Y-%m-%d %H:%M:%S%Z")

st.write("Now is ：", time)

st.markdown(
  """
    *:warning:实时地震快报：*
  """
)
# st.caption('（根据中国地震台网）')
st.info("注1： :blue[time为UTC时间,北京时间 = UTC + 8:00]")
st.info("注2： 由于streamlit cloud地址在国外,可能不能获取国内台网信息")
source = st.selectbox('数据源', (None, '美国地质局USGS', '中国地震台网'))

if source == '美国地质局USGS':
    url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv'
    df = pd.read_csv(url)
    df = df.head()
    st.dataframe(df)
    # st.dataframe(head)
    # 取出经纬度两列，然后传参数给 st.map()
    data2 = df.iloc[:, 1:3]
    st.map(data2, latitude='latitude', longitude='longitude', use_container_width=True)

# 中国地震台数据源：
if source == '中国地震台网':
    def extract_ceic_data():
        url = "https://news.ceic.ac.cn/index.html"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr')[:5]
            
            data = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    data.append(row_data)
            
            return pd.DataFrame(data) if data else None
        except Exception as e:
            st.error(f"获取数据时发生错误: {e}")
            return None
    
    df = extract_ceic_data()
    if df is not None:
        # 显示表格：
        st.dataframe(df)
        # 取出经纬度两列数据 (第2、3列)
        if len(df.columns) >= 4:
            data = df.iloc[:, 2:4]
            # 重命名列名
            data.columns = ['latitude', 'longitude']
            # 转换数据类型
            data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
            data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
            # 删除无效数据
            data = data.dropna()
            if not data.empty:
                st.map(data, use_container_width=True)
            else:
                st.warning("无法获取有效的地理位置数据")
        else:
            st.warning("数据格式不符合预期")
    else:
        st.error("无法获取中国地震台网数据")

st.divider()
st.write('更多地震数据，请进入 :point_down:')
st.page_link("pages/1数据获取.py")







