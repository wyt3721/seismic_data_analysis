import yaml
from yaml.loader import SafeLoader
import streamlit as st
import obspy
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import datetime
import streamlit_authenticator as stauth

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
    st.sidebar.write(f'你好， *{st.session_state["name"]}*')
    st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                           :email: wyt3721@outlook.com
                            Ver0.1a''')
    
    # authenticator.logout("退出", "sidebar")
 
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("## :earth_asia: 地震数据采集 :earth_asia:")
    st.divider()
    st.caption("地震数据包括地震事件目录和地震波形数据等")
    st.subheader("事件查询")
    options = st.selectbox("请选择数据中心：", ("IRIS", "USGS", "AUSPASS", "BGR", "EIDA", "EMSC", "ETH", "GEOFON", "GEONET", "GFZ",
                                               "ICGC", "IESDMC", "INGV",
                                               "IPGP", "IRISPH5", "ISC", "KNMI", "KOERI", "LMU", "NCEDC",
                                               "NIEP", "NOA", "ODC", "ORFEUS", "RASPISHAKE",
                                               "RESIF", "RESIFPH5", "SCEDC", "TEXNET", "UIB-NORSAR"))

    client = Client(options)

    st.write("请选择查询范围：", )

    col1, col2 = st.columns(2)
    with col1:
        t1 = st.date_input("开始日期", datetime.date(2024, 1, 1))
        # t3 = st.time_input('开始时间', datetime.time(0, 0), step=60)
        lon1 = st.slider(
            '经度范围：',
            -180.0, 180.0, (50.0, 150.0))
        min_lon = lon1[0]
        max_lon = lon1[1]
        # min_longitude = st.slider("最小经度：", min_value=-180.0, max_value=180.0, step=0.01, value=75.0)
        # min_latitude = st.slider("最小纬度：", min_value=-90.0, max_value=90.0, step=0.01, value=15.0)
        min_mag = st.slider("最小震级：", min_value=1.0, max_value=10.0, step=0.1, value=5.0)

    with col2:
        t2 = st.date_input("截止日期", datetime.datetime.now())
        # t4 = st.time_input('截止时间', datetime.time(23, 59), step=60)
        lat1 = st.slider(
            '纬度范围：',
            -90.0, 90.0, (10.0, 60.0))
        min_lat = lat1[0]
        max_lat = lat1[1]
        # max_longitude = st.slider("最大经度：", min_value=-180.0, max_value=180.0, step=0.01, value=150.0)
        # max_latitude = st.slider("最大纬度：", min_value=-90.0, max_value=90.0, step=0.01, value=60.0)
        min_depth = st.slider("最小深度(公里）：", min_value=10.0, max_value=100.0)


    @st.cache_data
    def get_cat(d1, d2, d3, d5, d7, d8, d9, d10):
        data = client.get_events(
            starttime=d1,
            endtime=d2,
            minmagnitude=d3,
            mindepth=d5,
            minlatitude=d7,
            maxlatitude=d8,
            minlongitude=d9,
            maxlongitude=d10,
        )
        return data


    cat = get_cat(t1, t2, min_mag, min_depth, min_lat, max_lat, min_lon, max_lon)
    cat_all = cat.__str__(print_all=True)

    if st.button("开始查询"):
        st.text(body=cat)

    st.download_button(label="下载目录", data=cat_all, file_name="catalog.csv", mime="text/csv")
    st.write(":point_down:")
    st.page_link("./pages/3地震分布.py", label="地震分布:globe_with_meridians:")

    st.divider()

    st.subheader("地震波形")

    t = st.text_input('请根据查询结果，输入地震时刻：（如2024-03-29T02:02:40.210000Z）', value="2024-04-02T23:58:11.228000Z")

    t = UTCDateTime(t)
    # client = Client("IRIS")

    if st.checkbox("显示一小时波形"):
        stt = client.get_waveforms("IU", "ANMO,AFI", "00", "LHZ", t, t + 60 * 60)
        pic = stt.plot()
        st.write("1小时波形图：", pic)

    client.get_waveforms("IU", "ANMO,AFI", "00", "LHZ", t, t + 60 * 60,
                             filename='uploads/waveform')
    with open('uploads/waveform', 'rb') as file:
        wave_data = file.read()
    st.download_button(label='下载波形', data=wave_data, file_name="download.mseed",mime="application/octet-stream")

    st.write(":point_down:")
    st.page_link('./pages/2波形分析.py', label="更多分析 :bell:")






