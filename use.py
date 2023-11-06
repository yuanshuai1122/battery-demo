from datetime import datetime

from joblib import load
import pandas as pd
from tensorflow.python.keras.saving.save import load_model

def time_to_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second

def str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h*3600 + m*60 + s


# 读取Excel文件
df = pd.read_excel('data/test.xlsx', sheet_name='具体数据系列2')

# 你的数据的表头是 ['步骤时间', '电流/A', '电压/V', '辅助温度/℃']
selected_headers = ['步骤时间', '电流/A', '电压/V', '辅助温度/℃']

# 创建一个空的列表来保存对象
objects = []

# 遍历DataFrame的每一行
for index, row in df.iterrows():
    # 提取所需的表头并创建一个对象
    obj = {header: row[header] for header in selected_headers}
    # 单独处理“步骤时间”
    obj['步骤时间'] = time_to_seconds(obj['步骤时间'])
    # 将对象添加到列表中
    objects.append(obj)


# 加载模型和scaler
model = load_model('./models/save_model.h5')
scaler = load('./scalers/scaler.joblib')

# 假设new_data是你要预测的新数据，它是一个dataframe
# new_data = pd.DataFrame({
#     '步骤时间': str_to_seconds("0:01:40"),
#     '电流/A': [15.002],
#     '电压/V': [3.9276],
#     '辅助温度/℃': [30.46]
# })
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