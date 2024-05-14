# import yaml
# from yaml.loader import SafeLoader
# import streamlit_authenticator as stauth
import streamlit as st
import time
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import sqlite3
import pandas as pd

# with open('./.streamlit/config.yaml') as file:
#     config = yaml.load(file, Loader=SafeLoader)


# authenticator = stauth.Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# if st.session_state.authentication_status is None:
#     st.switch_page('app.py')


# elif st.session_state["authentication_status"] is True:

st.header(':book:地震资料数据库')
st.divider()
st.write('查询您需要的地震资料:')
st.caption('云端暂不能使用数据库')
# st.sidebar.write(f'你好， *{st.session_state["name"]}*')
st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                       :email: wyt3721@outlook.com
                        Ver0.1a''')
# authenticator.logout("退出", "sidebar")
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.2)
#     my_bar.progress(percent_complete + 1)
s = st.selectbox('请选择数据库：', ['mysql', 'sqlite',  'postgresql'])

if s == 'sqlite':
    con = sqlite3.connect('pets.db')
    st.success('连接成功sqlite')
    cur = con.cursor()
    cur.execute("select * from test limit 10")
    res = cur.fetchall()
    res = pd.DataFrame(res)
    st.write(res)
    # engine = create_engine('env:url', echo=Ture)
    # engine
# 创建数据库连接
# conn = st.connection('mysql', 'sql')

# # 用sql语言查询数据库
# df = conn.query("select * from test limit 10;")
# st.dataframe(df)
