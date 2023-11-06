import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt


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
    print(f"开始删除某个文件夹/文件， folder_path:{folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print(f"删除某个文件夹/文件完成， folder_path:{folder_path}")


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
    last_part = pd.DataFrame()
    for i in indices:
        print(f"开始切分第{i}个文件.start:{start}")
        # 如果不是第一篇 需要跳过文字行
        if start != 0:
            start = start + 1
        df_part = df.iloc[start:i]
        df_part['测试时间'] = pd.to_numeric(df_part['测试时间'].apply(str_to_seconds), errors='coerce')
        df_part['步骤时间'] = pd.to_numeric(df_part['步骤时间'].apply(str_to_seconds), errors='coerce')

        # 获取符合条件的最后一行的步骤时间
        filtered_df_CCC = df_part[df_part['工步状态'] == 'CCC']
        last_operate_time = filtered_df_CCC.iloc[-1]['步骤时间']
        # 更新符合条件的某一列的值
        filtered_df_CVC = df_part[df_part['工步状态'] == 'CVC']
        filtered_df_CVC['步骤时间'] = filtered_df_CVC['步骤时间'] + last_operate_time
        # 将更新后的结果写入原始数据框
        df_part.update(filtered_df_CVC)

        # # 将步骤时间列的值覆盖到测试时间列的值
        # df['测试时间'] = df['步骤时间']

        df_part.to_excel(split_path + f'/split_{start}_{i}.xlsx', index=False)
        start = i
        last_part = df_part
    print(f'文件切分完毕，返回split_path:{split_path}')
    return split_path

# 画曲线图
def draw_img(folder_path, x_column, y_column):
    # 画图
    plt.figure(figsize=(10, 6))

    filenames = os.listdir(folder_path)
    # 遍历文件夹中的所有 Excel 文件
    for filename in filenames[::500]:
    #for filename in filenames:
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


if __name__ == '__main__':
#    new_file = handle_files('data/3号电池.xlsx')
#    path = split_file(new_file, 'data/split')
#     draw_img('data/split', '步骤时间', '容量/Ah')
#     draw_img('data/split', '步骤时间', '电流/A')
#    draw_img('data/split', '步骤时间', '电压/V')
    draw_img('data/split', '步骤时间', '辅助温度/℃')
    #draw_img('data/split', '电流/A')
    #print(get_file_last_line("data/split/split_0_34.xlsx"))


