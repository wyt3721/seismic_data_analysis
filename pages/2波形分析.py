import streamlit as st
import os
import obspy
import matplotlib.pyplot as plt
from obspy import read, Trace
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

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
#     st.switch_page('../app.py')


# else:
st.title(":bell: 地震波形文件处理 ")
st.divider()
st.caption('(支持 mseed/wav/sac  等格式)')
# st.sidebar.write(f'你好， *{st.session_state["name"]}*')
st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                       :email: wyt3721@outlook.com
                        Ver0.1a''')
# authenticator.logout("退出", "sidebar")
# 创建文件夹保存上传的文件
if not os.path.exists("uploads"):
    os.makedirs("uploads")

uploaded_file = st.file_uploader("请上传地震波形文件：")

    # 构造装饰器函数，避免刷新重复加载数据
@st.cache_data
def obspy_read(f):
    t = obspy.read(f)
    return t


if uploaded_file is not None:
    # 保存上传文件
    file_path = os.path.join("uploads/upload_waveforms")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.success(f"已保存文件在: {file_path}")

    # 读取文件：
    ob = obspy_read(uploaded_file)
    # st.text(ob)
    # 选择震道进行可视化

    tr = ob[0]
    st.write('波形元数据：')
    st.text(tr.stats)

# 绘制元数据地震道
    if st.button("震道信息"):
        trace = tr.plot(color='red')
        # trace = tr.plot(type='dayplot', color='green')
        st.write(ob)
        st.write(trace)

# 绘制地震道的振幅随时间的变化
    if st.button("振幅变化"):
        amp = tr.plot(type='amplitude', x='time', y='value', legend='amplitude', color='green')
        st.write(amp)

# 绘制地震道的频率内容
    if st.button("震道频率"):
        spec = tr.plot(type='spectrogram', color='blue')
        st.write(spec)

    if st.button("低通滤波"):
        ob.filter("lowpass", freq=0.1, corners=2)
        fil = ob.plot(interval=60, right_vertical_labels=False, vertical_scaling_range=5e3,
                      one_tick_per_line=True, show_y_UTC_label=False,
                      events={'min_magnitude': 5})
        st.write(fil)

    # TO DO
    # if st.button("下采样")：
    # if st.button("合并")：
    # if st.button("FK分析")：
    # if st.button("信号包络")：
    if st.button('触发器'):

        tri = tr.plot(type="relative")
        st.write(tri)



