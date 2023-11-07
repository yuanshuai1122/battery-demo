import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


# 带天数的字符串转秒 2-05:50:36
def str_to_seconds(time_str: str):
    time = time_str.split(":")
    if "-" in time[0]:
        split_day = time[0].split("-")
        return int(split_day[0]) * 86400 + int(split_day[1]) * 3600 + int(time[1]) * 60 + int(time[2])
    else:
        return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])


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
def handle_files(file_with_path: str):
    # 读取Excel文件
    print("开始读取excel:", file_with_path)
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

    # 处理新文件路径
    file_split = file_with_path.split('/')
    new_file_name = 'new_img_' + file_split[-1]
    new_file_path = "".join(file_split[:-1]) + "/" + new_file_name
    print("处理完毕, 原地址：", file_with_path, "新地址:", new_file_path)
    # 保存新excel
    df.to_excel(new_file_path, index=False)
    return new_file_path


# 分割文件
def split_file(file_with_path, split_path):
    # 清空文件夹
    delete_all_files_in_folder('data/split')
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
        df_part['测试时间'] = pd.to_numeric(df_part['测试时间'].apply(str_to_seconds), errors='coerce')
        df_part['步骤时间'] = pd.to_numeric(df_part['步骤时间'].apply(str_to_seconds), errors='coerce')
        # 删除测试时间列
        df_part = df_part.drop('测试时间', axis=1)
        # 获取符合条件的最后一行的步骤时间
        filtered_df_ccc = df_part[df_part['工步状态'] == 'CCC']
        last_operate_time = filtered_df_ccc.iloc[-1]['步骤时间']
        # 拷贝一列操作到 测试时间
        df_part['测试时间'] = df_part['步骤时间'].copy()
        # 更新符合条件的某一列的值
        filtered_df_cvc = df_part[df_part['工步状态'] == 'CVC']
        # 更新到测试时间
        filtered_df_cvc['测试时间'] = filtered_df_cvc['测试时间'].copy() + last_operate_time
        # 将更新后的结果写入原始数据框
        df_part.update(filtered_df_cvc)
        # 计算SOH到某SOH列
        last_capacity = df_part['容量/Ah'].iloc[-1]
        df_part['SOH'] = round(last_capacity / 28.243, 2)

        # 写入到excel
        df_part.to_excel(split_path + f'/第_{index + 1}_循环.xlsx', index=False)
        start = value

    print(f'文件切分完毕，返回split_path:{split_path}')
    return split_path


# 画曲线图
def draw_img(folder_path, x_column, y_column):
    # 画图
    plt.figure(figsize=(10, 6))

    # 遍历文件夹中的所有 Excel 文件
    filenames = os.listdir(folder_path)
    for filename in filenames:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print("出图前开始处理：", filename)
            # 读取 Excel 文件
            df = pd.read_excel(os.path.join(folder_path, filename))
            plt.plot(df[x_column], df[y_column], label=filename)

    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.title(x_column + '-' + y_column + "曲线")

    plt.show()


def draw_img_soh(folder_path, x_column):
    # y轴
    y_column = 'SOH'
    # 声明所有y值
    y_values = list()
    x_values = list()
    # 遍历文件夹中的所有 Excel 文件
    filenames = os.listdir(folder_path)
    for filename in filenames[::500]:
        # for filename in filenames:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print("SOH出图前开始处理：", filename)
            # 读取 Excel 文件
            df = pd.read_excel(os.path.join(folder_path, filename))
            y_values.append(df['SOH'].iloc[-1])
            x_values.append(filename.split('_')[1])

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('测试时间-SOH 曲线')

    plt.show()


# 数据合并
def combined_data(folder_path, target_path, circle_list):
    print(f'开始处理文件合并，folder_path:{folder_path}, target_path:{target_path}, circle_list:{circle_list}')
    delete_all_files_in_folder(target_path)
    # 定义一个空的DataFrame用于存储拼接后的数据
    combined_data = pd.DataFrame()
    # 遍历文件夹下的文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            if filename.split('_')[1] in circle_list:
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
    labels = df['SOH']  # 假设"SOH"是我们的目标变量

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
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks, verbose=0)

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

    # 保存模型
    model.save('./models/save_model.h5')

    # 保存scaler
    dump(scaler, './scalers/scaler.joblib')

# 获取训练数据集
def get_train_files(folder_path, move_path):

    # 清空要移动的目标文件夹
    delete_all_files_in_folder(move_path)

    # 定义一个空的DataFrame用于存储拼接后的数据
    capacity_list = []
    # 遍历文件夹下的文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            print('筛选文件:', filename)
            # 文件地址
            file_path = os.path.join(folder_path, filename)
            # 读取Excel文件内容
            df = pd.read_excel(file_path)
            last_capacity = df['容量/Ah'].iloc[-1]
            # 只保存大于70% DOD
            if last_capacity >= 22:
                capacity_list.append(filename.split('_')[1])
            else:
                shutil.move(file_path, move_path)

    return capacity_list


if __name__ == '__main__':
    # new_file = handle_files('data/3号电池.xlsx')
    # path = split_file(new_file, 'data/split')
    # draw_img('data/split', '测试时间', '容量/Ah')
    # draw_img('data/split', '测试时间', '电流/A')
    # draw_img('data/split', '测试时间', '电压/V')
    # draw_img('data/split', '测试时间', '辅助温度/℃')
    # draw_img_soh('data/split', '测试时间', 'SOH')
    #circle_list = get_train_files(path, 'data/dod_70/')
    circle_list = get_train_files('data/split', 'data/dod_70/')
    train_path = combined_data('data/split', 'data/train', circle_list)
    train_model(train_path)

if __name__ == '__main__':
    new_file = handle_files('data/4号电池.xlsx')
    path = split_file(new_file, 'data/split')
