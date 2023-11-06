import os
import pandas as pd
import matplotlib.pyplot as plt


# def str_to_seconds(time_str):
#     h, m, s = map(int, time_str.split(':'))
#     return h*3600 + m*60 + s

# 带天数的字符串转秒 2-05:50:36
def str_to_seconds(time_str: str):
    time = time_str.split(":")
    if "-" in time[0]:
        split_day = time[0].split("-")
        return int(split_day[0]) * 86400 + int(split_day[1]) * 3600 + int(time[1]) * 60 + int(time[2])
    else:
        return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])


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
    #df = df[(df['工步状态'] == 'CCC') | (df['工步状态'] == 'CVC') | (df['工步状态'] == '工步状态')]
    df = df[(df['工步状态'] == 'CCC') | (df['工步状态'] == '工步状态')]

    # 处理新文件路径
    file_split = file_with_path.split('/')
    new_file_name = 'new_img_' + file_split[-1]
    new_file_path = "".join(file_split[:-1]) + "/" + new_file_name
    print("处理完毕, 原地址：", file_with_path, "新地址:", new_file_path)
    # 保存新excel
    df.to_excel(new_file_path, index=False)
    return new_file_path

def split_file(file_with_path, split_path):

    # 读取文件
    df = pd.read_excel(file_with_path)

    # 获取第一列满足条件的行索引
    indices = df[df['工步状态'] == '工步状态'].index.tolist()
    print('获取excel切分索引，', indices)

    # 添加最后一行的索引
    indices.append(len(df))

    # 按照指定的行索引拆分DataFrame并保存为Excel文件
    start = 0
    for i in indices:
        print(f"开始切分第{i}个文件.start:{start}")
        # 如果不是第一篇 需要跳过文字行
        if start != 0:
            start = start + 1
        df_part = df.iloc[start:i]
        df_part.to_excel(split_path + f'/split_{start}_{i}.xlsx', index=False)
        start = i
    print(f'文件切分完毕，返回split_path:{split_path}')
    return split_path

# 画曲线图
def draw_img(folder_path, y_column):
    # 初始化一个空的 DataFrame 来保存所有文件的数据
    all_data = pd.DataFrame()

    filenames = os.listdir(folder_path)
    # 遍历文件夹中的所有 Excel 文件
    for filename in filenames[::500]:
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            print("出图前开始处理：", filename)
            # 读取 Excel 文件
            df = pd.read_excel(os.path.join(folder_path, filename))
            # 将数据添加到 all_data DataFrame
            all_data = pd.concat([all_data, df])

    all_data['测试时间'] = pd.to_numeric(all_data['测试时间'].apply(str_to_seconds), errors='coerce')
    all_data['步骤时间'] = pd.to_numeric(all_data['步骤时间'].apply(str_to_seconds), errors='coerce')
    all_data['电流/A'] = pd.to_numeric(all_data['电流/A'], errors='coerce')
    all_data['容量/Ah'] = pd.to_numeric(all_data['容量/Ah'], errors='coerce')
    all_data['电压/V'] = pd.to_numeric(all_data['电压/V'], errors='coerce')
    all_data['辅助温度/℃'] = pd.to_numeric(all_data['辅助温度/℃'], errors='coerce')

    all_data.to_excel('data/tmp/tmp.xlsx', index=False)


    # 指定横纵坐标为 Excel 的列
    x_column = '测试时间'

    # 中文显示
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    # plt.rcParams['font.size'] = 12  # 字体大小
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(all_data[x_column], all_data[y_column])

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(x_column + y_column + "曲线")

    plt.show()


if __name__ == '__main__':
    # new_file = handle_files('data/3号电池.xlsx')
    # path = split_file(new_file, 'data/split')
    # draw_img(path)
    draw_img('data/split', '辅助温度/℃')


