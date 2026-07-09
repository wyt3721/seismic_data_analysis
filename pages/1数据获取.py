import streamlit as st
import obspy
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import datetime
import io
import tempfile

# ================= 页面布局与侧边栏 =================
st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                   :email: wyt3721@outlook.com
                    Ver0.1a''')

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.markdown("## :earth_asia: 地震数据采集 :earth_asia:")
st.divider()
st.caption("地震数据包括地震事件目录和地震波形数据等")

# ================= 事件查询部分 =================
st.subheader("事件查询")
options = st.selectbox("请选择数据中心：", (
    "IRIS", "USGS", "AUSPASS", "BGR", "EIDA", "EMSC", "ETH", "GEOFON", "GEONET", "GFZ",
    "ICGC", "IESDMC", "INGV", "IPGP", "IRISPH5", "ISC", "KNMI", "KOERI", "LMU", "NCEDC",
    "NIEP", "NOA", "ODC", "ORFEUS", "RASPISHAKE", "RESIF", "RESIFPH5", "SCEDC", "TEXNET", "UIB-NORSAR"
))

# 注意：将 Client 初始化移出缓存函数，避免缓存哈希冲突
client = Client(options)

col1, col2 = st.columns(2)
with col1:
    t1 = st.date_input("开始日期", datetime.date(2024, 1, 1))
    lon_range = st.slider('经度范围：', -180.0, 180.0, (73.0, 135.0))
    min_longitude, max_longitude = lon_range
    min_mag = st.slider("最小震级：", min_value=1.0, max_value=10.0, step=0.1, value=5.0)

with col2:
    t2 = st.date_input("截止日期", datetime.datetime.now())
    lat_range = st.slider('纬度范围：', -90.0, 90.0, (4.0, 53.0))
    min_latitude, max_latitude = lat_range
    min_depth = st.slider("最小深度(公里）：", min_value=5.0, max_value=100.0)

# 优化缓存：使用 _client 忽略对象哈希，增加异常捕获防止 ValueError 崩溃
@st.cache_data(show_spinner="正在从数据中心查询事件目录...")
def get_cat(_client, d1, d2, d3, d5, lat_min, lat_max, lon_min, lon_max):
    try:
        data = _client.get_events(
            starttime=d1, endtime=d2, minmagnitude=d3, mindepth=d5,
            minlatitude=lat_min, maxlatitude=lat_max,
            minlongitude=lon_min, maxlongitude=lon_max,
        )
        return data
    except ValueError:
        # 捕获 ObsPy 查不到数据或参数越界的异常
        return None
    except Exception as e:
        st.error(f"查询过程中发生网络或服务错误: {e}")
        return None

# 调用函数
cat = get_cat(client, t1, t2, min_mag, min_depth, min_latitude, max_latitude, min_longitude, max_longitude)

# 处理查询结果
if cat is None or len(cat) == 0:
    st.warning("⚠️ 未查询到符合条件的地震事件，请尝试放宽查询条件（如降低震级、扩大经纬度范围或延长日期）。")
else:
    if st.button("开始查询"):
        st.text(body=str(cat))

    # 使用内存缓冲区导出真正的 CSV
    buffer = io.BytesIO()
    cat.write(buffer, format="CSV")
    buffer.seek(0)
    st.download_button(label="下载目录", data=buffer, file_name="catalog.csv", mime="text/csv")

st.write(":point_down:")
st.page_link("./pages/3地震分布.py", label="地震分布:globe_with_meridians:")
st.divider()

# ================= 地震波形部分 =================
st.subheader("地震波形")
t = st.text_input('请根据查询结果，输入地震时刻：（如2024-03-29T20:00:00.000000Z）', value="2024-04-02T23:58:11.228000Z")
t = UTCDateTime(t)

if st.checkbox("显示一小时波形"):
    try:
        # 获取波形数据
        stt = client.get_waveforms("IU", "ANMO,AFI", "00", "LHZ", t, t + 60 * 60)
        
        # 适配新版 ObsPy：传入 show=False 获取 figure 对象
        fig = stt.plot(show=False) 
        st.pyplot(fig)

        # 将下载逻辑放入条件块，并使用临时文件避免本地路径报错
        with tempfile.NamedTemporaryFile(suffix=".mseed", delete=False) as tmp:
            stt.write(tmp.name, format="MSEED")
            with open(tmp.name, 'rb') as f:
                wave_data = f.read()
                
        st.download_button(label='下载波形', data=wave_data, file_name="download.mseed", mime="application/octet-stream")
        
    except Exception as e:
        st.error(f"波形获取失败，请检查时间或台站代码是否正确。错误信息：{e}")

st.write(":point_down:")
st.page_link('./pages/2波形分析.py', label='更多分析 :bell:')
