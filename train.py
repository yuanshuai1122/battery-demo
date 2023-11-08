import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# 带天数的字符串转秒 2-05:50:36
def str_to_seconds(time_str: str):
    # 不包含时间格式直接返回
    if ':' in time_str:
        time = time_str.split(":")
        if "-" in time[0]:
            split_day = time[0].split("-")
            return int(split_day[0]) * 86400 + int(split_day[1]) * 3600 + int(time[1]) * 60 + int(time[2])
        else:
            return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
    else:
        return time_str


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


# 文件处理
def handle_files(file_with_path: str, new_folder):
    # 读取Excel文件
    print(f"开始读取excel:{file_with_path}, new_folder:{new_folder}")
    #delete_all_files_in_folder(new_folder)

    df = pd.read_excel(file_with_path, sheet_name='具体数据系列2', header=None)
    # 删除第一行（原的表头）
    df = df.iloc[1:]

    # 将第一行的值设置为列名
    df.columns = df.iloc[0]

    # 删除现在的第一行（因为它已经被用作列名）
    df = df.iloc[1:]
    # 删除空行
    df = df.dropna(how='all')
    # 只保留需要的数据
    df = df[['测试时间', '步骤时间', '电流/A', '容量/Ah', '电压/V', '辅助温度/℃', '工步状态']]
    # 保留符合条件的行
    df = df[(df['工步状态'] == 'CCC') | (df['工步状态'] == 'CVC') | (df['工步状态'] == '工步状态')]
    df['步骤时间'] = pd.to_numeric(df['步骤时间'].astype(str).apply(str_to_seconds), errors='coerce')
    # 处理新文件路径
    file_split = file_with_path.split('/')
    new_file_name = 'new_origin_' + file_split[-1]
    new_file_path = new_folder + "/" + new_file_name
    print("处理完毕, 原地址：", file_with_path, "新地址:", new_file_path)
    # 保存新excel
    df.to_excel(new_file_path, index=False)
    return new_file_path


# 分割文件
def split_file(file_with_path, split_path, init_capacity):
    # 清空文件夹
    delete_all_files_in_folder(split_path)
    # 读取文件
    df = pd.read_excel(file_with_path)

    # 获取第一列满足条件的行索引
    indices = df[df['工步状态'] == '工步状态'].index.tolist()
    print('获取excel切分索引，', indices)

    # 添加最后一行的索引
    indices.append(len(df))

    # 按照指定的行索引拆分DataFrame并保存为Excel文件
    start = 0
    for index, value in enumerate(indices):
        print(f"开始切分第{index + 1}个文件.start:{start}")
        # 如果不是第一篇 需要跳过文字行
        if start != 0:
            start = start + 1
        df_part = df.iloc[start:value].copy()
        # 删除测试时间列
        df_part = df_part.drop('测试时间', axis=1)
        # 获取符合条件的最后一行的步骤时间
        filtered_df_ccc = df_part[df_part['工步状态'] == 'CCC']
        last_operate_time = filtered_df_ccc.iloc[-1]['步骤时间']
        # 拷贝一列操作到 测试时间
        df_part['测试时间'] = df_part['步骤时间']
        # 更新符合条件的某一列的值
        filtered_df_cvc = df_part[df_part['工步状态'] == 'CVC']
        # 更新到测试时间
        #filtered_df_cvc['测试时间'] = filtered_df_cvc['测试时间'].copy() + last_operate_time
        # 更新到测试时间
        filtered_df_cvc.loc[:, '测试时间'] = filtered_df_cvc.loc[:, '测试时间'].apply(lambda x: x + last_operate_time)
        # 将更新后的结果写入原始数据框
        df_part.update(filtered_df_cvc)
        # 更新步数
        start = value
        # 计算SOH到某SOH列
        last_capacity = df_part['容量/Ah'].iloc[-1]
        # 判断是否是70DOD 需要跳过
        if last_capacity < init_capacity * 0.8:
            continue

        # 计算SOH
        df_part['SOH'] = round(last_capacity / init_capacity, 2)
        # 写入到excel
        df_part.to_excel(split_path + f'/第_{index + 1}_循环.xlsx', index=False)
        start = value

    print(f'文件切分完毕，返回split_path:{split_path}')
    return split_path


# 画曲线图
def draw_img(folder_path, x_column, y_column, title):
    # 画图
    plt.figure(figsize=(10, 6))
    # 遍历文件夹中的所有 Excel 文件
    filenames = os.listdir(folder_path)
    # 根据文件名中的数字进行排序
    sorted_filenames = sorted(filenames[::10], key=lambda x: int(x.split('_')[1]))
    num_files = len(sorted_filenames)
    cmap = cm.get_cmap('winter', num_files)  # 使用viridis颜色映射，根据文件数量生成颜色渐变

    for i, filename in enumerate(sorted_filenames):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print("出图前开始处理：", filename)
            # 读取 Excel 文件
            df = pd.read_excel(os.path.join(folder_path, filename))
            plt.plot(df[x_column], df[y_column], label=filename.split('.')[0], linewidth=2, color=cmap(i))

    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.title(title + " " + x_column + '-' + y_column + "曲线")

    plt.show()


def moving_average(data, window_size):
    # 创建一个窗口，窗口大小为window_size
    window = np.ones(window_size) / window_size
    # 使用np.convolve函数进行卷积操作，对数据进行平滑处理
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

def draw_img_soh(folder_path, x_column, title):
    # y轴
    y_column = 'SOH'
    # 声明所有y值
    y_values = list()
    x_values = list()
    # 遍历文件夹中的所有 Excel 文件
    filenames = os.listdir(folder_path)
    # 根据文件名中的数字进行排序
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[1]))
    for filename in sorted_filenames:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print("SOH出图前开始处理：", filename)
            # 读取 Excel 文件
            df = pd.read_excel(os.path.join(folder_path, filename))
            y_values.append(df['SOH'].iloc[-1])
            x_values.append(filename.split('_')[1])

    # 平滑处理
    window_size = 5  # 窗口大小
    smoothed_y_values = moving_average(y_values, window_size)

    # 去除开始和结束部分
    start_index = 10  # 开始索引
    end_index = -10  # 结束索引
    x_values = x_values[start_index:end_index]
    smoothed_y_values = smoothed_y_values[start_index:end_index]

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, smoothed_y_values, linewidth=2)  # 使用平滑后的 y 值，并增加线条宽度
    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(title + ' ' + x_column + '-SOH 曲线')

    # 设置横坐标刻度间距
    x_ticks = np.arange(0, len(x_values), 10)  # 每隔5个数据点设置一个刻度
    plt.xticks(x_ticks)

    plt.show()


# 数据合并
def combined_data(folder_path, target_path):
    print(f'开始处理文件合并，folder_path:{folder_path}, target_path:{target_path}')
    # 定义一个空的DataFrame用于存储拼接后的数据
    combined_data = pd.DataFrame()
    # 遍历文件夹下的文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            print('合并处理文件:', filename)
            # 文件地址
            file_path = os.path.join(folder_path, filename)
            # 读取Excel文件内容
            df = pd.read_excel(file_path)
            # 将数据拼接到combined_data中
            combined_data = pd.concat([combined_data, df], ignore_index=True)


    target_path = target_path + "/train_data.xlsx"
    # 将拼接后的数据保存到新的Excel文件中
    combined_data.to_excel(target_path, index=False)

    return target_path


# 训练模型
def train_model(file_with_path):
    # 读取Excel文件
    print("开始读取excel数据集:", file_with_path)
    df = pd.read_excel(file_with_path)  # 请将'your_file.xlsx'替换为你的Excel文件的路径
    # 填充缺失值
    # df.fillna(df.mean(), inplace=True)

    # 选择需要的参数
    features = df[['步骤时间', '电流/A', '电压/V', '辅助温度/℃']]
    labels = df['SOH']  # "SOH"是我们的目标变量

    print("数据集处理完毕")

    # 将数据分割成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 标准化数据
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 将数据重塑为CNN模型所需的形状
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # 创建模型
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 添加EarlyStopping和ModelCheckpoint回调函数：EarlyStopping用于提前停止训练，ModelCheckpoint用于保存模型
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('./models/model.h5', save_best_only=True)
    ]

    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks, verbose=0)

    # 打印模型总结
    model.summary()

    # 评估模型
    loss = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)

    # 查看其他评估指标
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print('Test MAE:', mae)
    print('Test R^2:', r2)

    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # 绘制损失值变化图
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('损失值变化')
    plt.legend()
    plt.show()

    # 保存模型
    model.save('./models/save_model.h5')

    # 保存scaler
    dump(scaler, './scalers/scaler.joblib')


def handle_all_files():
    new_file = handle_files('data/origin/4号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_4', 28.785)
    print(path)

    new_file = handle_files('data/origin/5号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_5', 28.573)
    print(path)

    new_file = handle_files('data/origin/6号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_6', 28.117)
    print(path)

    new_file = handle_files('data/origin/7号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_7', 28.590)
    print(path)

    new_file = handle_files('data/origin/8号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_8', 28.318)
    print(path)

    new_file = handle_files('data/origin/9号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_9', 28.096)
    print(path)

    new_file = handle_files('data/origin/10号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_10', 28.546)
    print(path)

    new_file = handle_files('data/origin/11号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_11', 28.394)
    print(path)

    new_file = handle_files('data/origin/12号电池.xlsx', 'data/transfer')
    print(new_file)
    path = split_file(new_file, 'data/split_12', 28.690)
    print(path)



if __name__ == '__main__':
    # new_file = handle_files('data/origin/3号电池.xlsx', 'data/transfer')
    # print(new_file)
    # path = split_file(new_file, 'data/split', 28.243)
    # print(path)
    #draw_img('data/split', '测试时间', '电流/A', '3号电池')
    #draw_img('data/split', '测试时间', '电压/V', '3号电池')
    #draw_img('data/split', '测试时间', '辅助温度/℃', '3号电池')
    #draw_img_soh('data/split', '循环次数', '3号电池')

    #train_path = combined_data('data/split', 'data/train')
    train_model('data/train/train_data.xlsx')
    #handle_all_files()



