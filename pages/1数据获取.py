import streamlit as st
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

col1, col2 = st.columns(2)

with col1:
    t1 = st.date_input("开始日期", datetime.date(2024, 1, 1))
    lon_range = st.slider('经度范围：', -180.0, 180.0, (73.0, 135.0))
    min_longitude, max_longitude = lon_range
    min_mag = st.slider("最小震级：", min_value=1.0, max_value=10.0, step=0.1, value=3.0)

with col2:
    t2 = st.date_input("截止日期", datetime.datetime.now())
    lat_range = st.slider('纬度范围：', -90.0, 90.0, (4.0, 53.0))
    min_latitude, max_latitude = lat_range
    min_depth = st.slider("最小深度(公里）：", min_value=0.0, max_value=700.0, value=0.0)

# 查询按钮（必须顶格写）
if st.button("🔍 开始查询", type="primary"):
    # 初始化 Client
    client = Client(options)
    
    # 【修复】不再使用 has_service()，而是直接尝试查询并捕获异常
    with st.spinner(f"正在从 {options} 查询地震目录，请稍候..."):
        try:
            cat = client.get_events(
                starttime=t1,
                endtime=t2,
                minmagnitude=min_mag,
                mindepth=min_depth,
                minlatitude=min_latitude,
                maxlatitude=max_latitude,
                minlongitude=min_longitude,
                maxlongitude=max_longitude,
            )
            
            if cat is None or len(cat) == 0:
                st.warning("⚠️ 未查询到符合条件的地震事件，请尝试放宽查询条件。")
            else:
                st.success(f"✅ 成功查询到 {len(cat)} 条地震记录！")
                st.text(body=str(cat))

                # 提供 JSON 格式下载
                buffer = io.BytesIO()
                json_str = cat.write(buffer,format="JSON")
                buffer.write(json_str.encode('utf-8'))
                buffer.seek(0)
                
                st.download_button(label="📥 下载目录 (JSON)", data=buffer, file_name="catalog.json", mime="application/json")
                
        except ValueError as e:
            st.error(f"❌ 参数错误或无匹配数据: {e}")
        except Exception as e:
            # 捕获所有异常，如果是数据中心不支持事件查询，会在这里被捕获并给出友好提示
            error_msg = str(e)
            if "does not have an event service" in error_msg:
                st.error(f"❌ 数据中心 [{options}] 不支持地震事件目录查询！请选择 IRIS、USGS 或 EMSC 等支持该服务的中心。")
            else:
                st.error(f"❌ 查询过程中发生网络或服务错误: {e}")

st.divider()

# ================= 地震波形部分 =================
st.subheader("地震波形")

t = st.text_input('请根据查询结果，输入地震时刻：（如2024-03-29T20:00:00.000000Z）', value="2024-04-02T23:58:11.228000Z")
t = UTCDateTime(t)

if st.checkbox("显示一小时波形"):
    try:
        # 注意：如果未点击查询，client 未定义，这里会报错。
        # 生产环境建议将 client 初始化提取到按钮外部或 session_state
        if 'client' not in locals():
            client = Client(options)
            
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
                
        st.download_button(label='📥 下载波形 (MSEED)', data=wave_data, file_name="download.mseed", mime="application/octet-stream")
        
    except Exception as e:
        st.error(f"波形获取失败，请检查时间或台站代码是否正确。错误信息：{e}")

st.write(":point_down:")
st.page_link('./pages/2波形分析.py', label='更多分析 :bell:')
