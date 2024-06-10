import streamlit as st
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from pyspark.sql.functions import to_timestamp, col
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_date, year, month, dayofmonth, hour, minute, second
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import unix_timestamp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,ElasticNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# import seaborn as sn
from sklearn.linear_model import  Ridge
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.svm import  SVR
from sklearn.tree import  DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures

st.title(":game_die: 机器学习 ")
st.divider()


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
    data = pd.read_csv(file_path)  # 读取csv
    des = st.radio('数据集概览', ['前五项', '信息', '统计'])
    if des == '前五项':
        st.dataframe(data.head())  # 查看前五行
    if des == '信息':
        st.text(data.columns.tolist())
    # data.tail()
    # data.sample()
    # data.info()  # 查看数据的类型，完整性
    # data.describe()  # 查看数据的统计特征（均值、方差等）
    data.dropna(inplace=True)  # 删除有缺失的样本

    # st.sidebar.success('')
    tr = st.sidebar.selectbox( ' 选择计算框架 ',('None', 'Spark', 'pytorch', 'Scikit-learn'))
    if tr == 'Spark':
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
        alg = st.selectbox('请选择学习算法', ('None', '线性回归', '随机森林', 'XGboost', 'others'))
        if alg == '线性回归':
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

    if tr == 'Scikit-learn':
        # st.success('Scikit-learn')

        # for id in data.columns[:-1]:
        #     fig = sn.pairplot(data[[id, data.columns[-1]]])
        #     st.write(fig)
        y = data['MEDV']  # 标签-房价
        X = data.drop(['MEDV'], axis=1)  # 去掉标签（房价）的数据子集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        st_agg = st.sidebar.selectbox('选择算法测试',('lr','ElasticNet','lasso', 'KN', 'Ridge', 'SVR', 'DTR', 'PolyFeature'))
        if st_agg == 'lr':
            lr = LinearRegression()  # 实例化一个线性回归对象
            lr.fit(X_train, y_train)  # 采用fit方法，拟合回归系数和截距
            # print(lr.intercept_)  # 输出截距
            # print(lr.coef_)  # 输出系数   可分析特征的重要性以及与目标的关系
            y_pred = lr.predict(X_test)  # 模型预测
            st.write("R2=", r2_score(y_test, y_pred))  # 模型评价, 决定系数
            st.write("mse=",mean_squared_error(y_test, y_pred))#均方误差
            # print(lr.intercept_)  #输出截距
            # print(lr.coef_)  #系数
            plt.plot(y_test.values, c="r", label="y_test")
            plt.plot(y_pred, c="b", label="y_pred")
            plt.legend()
            fig = plt.show()
            st.write(fig)
        if st_agg == 'ElasticNet':
            EN = ElasticNet(0.01)  # 实例化弹性网络回归对象
            EN.fit(X_train, y_train)  # 训练
            y_pred = EN.predict(X_test)  # 预测
            # 评价
            st.write(r2_score(y_pred, y_test))
            # print("mse=",mean_squared_error(y_test, y_pred))#均方误差
            y_predt = EN.predict(X_train)  # 查看训练集上的效果
            st.write(r2_score(y_predt, y_train))
        if st_agg == 'lasso':
            la = Lasso()
            la.fit(X_train, y_train)  # 拟合
            y_pred = la.predict(X_test)  # 预测
            # 评价
            st.write(r2_score(y_pred, y_test))
            # print("mse=",mean_squared_error(y_test, y_pred))#均方误差
            y_predt = la.predict(X_train)  # 查看训练集上的效果
            st.write(r2_score(y_predt, y_train))
            # prtin(la.coef_)   #输出系数 （部分系数为“0”，lasso常用与特征提取）  可分析特征的重要性以及与目标的关系
        if st_agg == 'Ridge':
            rd = Ridge(0.01)
            rd.fit(X_train, y_train)
            y_pred = rd.predict(X_test)
            st.write(r2_score(y_pred, y_test))
            y_predt = rd.predict(X_train)
            st.write(r2_score(y_predt, y_train))
        if st_agg == 'KN':
            Knr = KNeighborsRegressor()
            Knr.fit(X_train, y_train)
            y_pred = Knr.predict(X_test)
            st.write(r2_score(y_pred, y_test))
            y_predt = Knr.predict(X_train)
            r2_score(y_predt, y_train)
        if st_agg =='SVR':
            svr = SVR()
            svr.fit(X_train, y_train)
            y_pred = svr.predict(X_test)
            st.write(r2_score(y_pred, y_test))
            y_predt = svr.predict(X_train)
            r2_score(y_predt, y_train)
        if st_agg == 'DTR':
            dtr = DecisionTreeRegressor(max_depth=4)
            dtr.fit(X_train, y_train)
            y_pred = dtr.predict(X_test)
            st.write(r2_score(y_pred, y_test))
            y_predt = dtr.predict(X_train)
            st.write(r2_score(y_predt, y_train))
        if st_agg == 'PolyFeature':
            poly = PolynomialFeatures(degree=2)  # 添加特征(升维)
            poly.fit(X_train)
            poly.fit(X_test)
            X_1 = poly.transform(X_train)
            X_2 = poly.transform(X_test)
            # 训练
            lin_reg = LinearRegression()
            lin_reg.fit(X_1, y_train)
            # 预测、评价
            y_pred = lin_reg.predict(X_1)
            st.write(r2_score(y_pred, y_train))
            y_pred = lin_reg.predict(X_2)
            st.write(r2_score(y_pred, y_test))
    if tr == 'pytorch':
        # ----------数据集----------

        # MNIST数据集示例
        # training_data = datasets.FashionMNIST(
        #     root="data",
        #     train=True,
        #     download=True,
        #     transform=ToTensor(),
        # )
        # # 加载MNIST数据集的测试集
        # test_data = datasets.FashionMNIST(
        #     root="data",
        #     train=False,
        #     download=True,
        #     transform=ToTensor(),
        # )
        # training_data =
        # test_data =


        # # batch大小
        batch_size = 64

        # 创建dataloader
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # 遍历dataloader
        for X, y in test_dataloader:
            print("Shape of X [N, C, H, W]: ", X.shape)  # 每个batch数据的形状
            print("Shape of y: ", y.shape)  # 每个batch标签的形状
            break


        # ----------模型----------
        # 定义模型
        class NeuralNetwork(nn.Module):
            def __init__(self):  # 初始化，实例化模型的时候就会调用
                super(NeuralNetwork, self).__init__()
                self.flatten = nn.Flatten()  # [64, 1, 28, 28] -> [64, 1*28*28]
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(28 * 28, 512),  # [64, 1*28*28] -> [64, 512]
                    nn.ReLU(),
                    nn.Linear(512, 512),  # [64, 512] -> [64, 512]
                    nn.ReLU(),
                    nn.Linear(512, 10)  # [64, 512] -> [64, 10]
                )

            def forward(self, x):  # 前向传播，输入数据进网络的时候才会调用
                x = self.flatten(x)  # [64, 1*28*28]
                logits = self.linear_relu_stack(x)  # [64, 10]
                return logits


        # 使用gpu或者cpu进行训练
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 打印使用的是gpu/cpu
        st.write("Using {} device".format(device))
        # 实例化模型
        model = NeuralNetwork().to(device)
        # 打印模型结构
        st.write(model)







model_url = st.text_input('请输入模型地址：')


