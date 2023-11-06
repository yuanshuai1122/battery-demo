import openpyxl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense


def time_to_seconds(time):
    return (time.hour * 60 + time.minute) * 60 + time.second

# 读取Excel文件
print("开始读取excel数据集...")
df = pd.read_excel('data/test.xlsx', sheet_name='具体数据系列2')  # 请将'your_file.xlsx'替换为你的Excel文件的路径

# 填充缺失值
# df.fillna(df.mean(), inplace=True)

# 选择需要的参数
features = df[['步骤时间', '电流/A', '电压/V', '辅助温度/℃']]
# 转换步骤时间列
features['步骤时间'] = features['步骤时间'].apply(time_to_seconds)
labels = df['容量/Ah']  # 假设"容量/Ah"是我们的目标变量

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

