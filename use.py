import os
import shutil
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

# 删除某个文件或者某个文件夹下所有文件
def delete_all_files_in_folder(folder_path):
    print(f"开始删除文件夹/文件， folder_path:{folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print(f"删除文件夹/文件完成， folder_path:{folder_path}")


def use_model(folder_path, model, scaler):
    print(f"开始应用模型, folder_path:{folder_path}, model:{model}, scaler:{scaler}")
    # 读取 Excel 文件
    df = pd.read_excel(folder_path)
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

def handle_file(origin_path_data, to_path,  columns):

    delete_all_files_in_folder(to_path)

    df = pd.read_csv(origin_path_data)
    df = df.sort_values(by='soc', ascending=True)
    # 删除空行
    df = df.dropna(how='all')
    # 只保留需要的数据
    df = df[['current', 'voltages', 'soc', 'receive_time']]
    df['current'] = df['current'].apply(lambda x: -x)
    df['voltages'] = df['voltages'].apply(lambda x: x.split('[')[1].split(']')[0])
    # 使用 str.split() 方法拆分列
    df[columns] = df['voltages'].str.split(', ', expand=True)
    # 删除原始列
    df.drop('voltages', axis=1, inplace=True)

    # 将日期时间列解析为 Pandas 的日期时间类型
    df['receive_time'] = pd.to_datetime(df['receive_time'])
    # 计算日期时间列的秒数
    df['seconds'] = (df['receive_time'] - df['receive_time'].min()).dt.total_seconds()
    # 归一化秒数，从0开始
    #df['测试时间'] = (df['seconds'] - df['seconds'].min()) / (df['seconds'].max() - df['seconds'].min())
    df['测试时间'] = (df['seconds'] - df['seconds'].min())

    for col in columns:
        df_part = df[['current', col, 'soc', '测试时间']]
        df_part = df_part.rename(columns={col: '电压/V', 'current': '电流/A'})
        df_part.to_excel(to_path + '/' + col + '.xlsx', index=False)

    return to_path


if __name__ == '__main__':

    cols = ['voltage_1', 'voltage_2', 'voltage_3', 'voltage_4', 'voltage_5', 'voltage_6', 'voltage_7', 'voltage_8', 'voltage_9', 'voltage_10', 'voltage_11',
        'voltage_12', 'voltage_13', 'voltage_14', 'voltage_15', 'voltage_16', 'voltage_17']

    folder_path = 'data/use/split'

    file_path = handle_file('data/use/Result_10.csv', folder_path, cols)


    result_list = list()
    # 遍历文件夹中的所有 Excel 文件
    filenames = os.listdir(folder_path)
    # 根据文件名中的数字进行排序
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in sorted_filenames:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            result = use_model(os.path.join(folder_path, filename), 'models/model.h5', 'scalers/scaler.joblib')
            result_list.append(result)

    print(result_list)











