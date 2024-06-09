import streamlit as st
# import streamlit_authenticator as stauth
# import yaml
# from yaml.loader import SafeLoader
import os
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
# from pyspark.ml.feature import VectorAssembler
# from pyspark.sql.functions import to_date, year, month, dayofmonth, hour, minute, second
from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import LinearRegression
# from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import unix_timestamp


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

# else:
st.title(":game_die: 机器学习预测 ")
st.divider()
# st.sidebar.write(f'你好， *{st.session_state["name"]}*')
st.sidebar.success('''欢迎使用本系统 :smile: :joy:
                       :email: wyt3721@outlook.com
                        Ver0.1a''')
# authenticator.logout("退出", "sidebar")
# 创建一个文件夹用于保存上传的文件
if not os.path.exists("learning"):
    os.makedirs("learning")

uploaded_file = st.file_uploader("请上传数据集：")
# 保存文件
if uploaded_file is not None:
    file_path = os.path.join("./upload_dataset")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.success(f"已保存数据集在: {file_path}")
    # 开启spark会话
    spark = SparkSession.builder \
        .appName("Earthquake Prediction") \
        .getOrCreate()
    st.success('Spark连接成功！')
   
    # 加载数据集
    data = spark.read.csv(file_path, header=True, inferSchema=True)
    # data.head(10)

    # data.describe()
    # 数据预处理
    st.selectbox("请选择特征：", ('time', 'latitude', 'longitude', 'depth'))
    data = data.withColumn("time_numeric", unix_timestamp(col("time")))

    assembler = VectorAssembler(inputCols=["time_numeric", "latitude", "longitude", "depth"], outputCol="features")
    output = assembler.transform(data)
    train_data, test_data = output.select("features", "mag").randomSplit([0.7, 0.3], seed=42)

    # 构建和训练模型
    st.selectbox('请选择学习算法', ('线性回归','随机森林', 'XGboost', 'others'))
    lr = LinearRegression(featuresCol="features", labelCol="mag")
    pipeline = Pipeline(stages=[lr])
    model = pipeline.fit(train_data)

    # 评估模型
    st.selectbox('选择评估方法', ('二分', '回归', '多分类', '多标签', '聚类', '排序'))
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="mag", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    st.write("均方根误差是 ： %g" % rmse)

    # 保存模型
    st.write('保存模型')
    # model.write().overwrite().save("path/to/save/model")
    st.write('加载模型')
    # 关闭SparkSession
    spark.stop()
  
model_url = st.text_input('请输入模型地址：')


