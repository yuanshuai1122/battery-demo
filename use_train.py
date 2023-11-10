import os
from datetime import datetime

import numpy as np
from joblib import load
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.saving.save import load_model
import random

# 设置全局字体
plt.rcParams['font.family'] = 'Arial Unicode MS'

def time_to_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second

def str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s


def use_model(origin_path_data, model, scaler):
    # 读取Excel文件
    df = pd.read_excel(origin_path_data)
    # 归一化秒数，从0开始
    #df['测试时间'] = (df['测试时间'] - df['测试时间'].min()) / (df['测试时间'].max() - df['测试时间'].min())
    selected_headers = ['测试时间', '电流/A', '电压/V']
    # 创建一个空的列表来保存对象
    objects = []
    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 提取所需的表头并创建一个对象
        obj = {header: row[header] for header in selected_headers}
        # 将对象添加到列表中
        objects.append(obj)
    # 加载模型和scaler
    model = load_model(model)
    scaler = load(scaler)
    new_data = pd.DataFrame(objects)
    # 填充缺失值
    new_data.fillna(new_data.mean(), inplace=True)
    # 使用相同的scaler进行标准化
    new_data = scaler.transform(new_data)
    # 重塑数据到正确的形状
    new_data = new_data.reshape((new_data.shape[0], new_data.shape[1], 1))
    # 使用模型进行预测
    prediction = model.predict(new_data)
    # 打印预测结果
    print('Predicted SOH:', prediction[0][0])
    return prediction[0][0]


def get_random_file(folder_path):
    files = os.listdir(folder_path)
    random_file = random.choice(files)
    return folder_path + '/' + random_file

def plot_distribution(data):
    # 创建直方图
    plt.hist(data, bins=30, density=True, alpha=0.5, color='blue')
    plt.xlabel('数值')
    plt.ylabel('频率')
    plt.title('SOH分布直方图')

    # 创建密度图
    plt.figure()
    plt.plot(data, np.zeros_like(data), 'b+', markersize=2)
    plt.xlabel('数值')
    plt.ylabel('密度')
    plt.title('SOH分布密度图')

    # 显示图表
    plt.show()


if __name__ == '__main__':

    total_values = []

    for i in range(20):
        print("迭代次数:", i + 1)
        total_values.append(use_model(get_random_file('data/split'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_4'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_5'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_6'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_7'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_8'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_9'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_10'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_11'), 'models/save_model.h5', 'scalers/scaler.joblib'))
        total_values.append(use_model(get_random_file('data/split_12'), 'models/save_model.h5', 'scalers/scaler.joblib'))

    print(total_values)

    plot_distribution(total_values)

    # use_model('data/split_7/第_1720_循环.xlsx', 'models/model.h5', 'scalers/scaler.joblib')
    # use_model('data/split_7/第_4_循环.xlsx', 'models/model.h5', 'scalers/scaler.joblib')
    # use_model('data/split_7/第_3227_循环.xlsx', 'models/model.h5', 'scalers/scaler.joblib')











