# 一 前言



## SOC 的预估或计算

电量状态（State of charge）（充电程度，剩余电量）是指当前电池内所含电量，以百分比表示。100%即电池完全充满；反之，0%即电池完全放电。

间接估测电量状态的方式一般可以分为以下五种：
* 化学反应前后之物质量
* 电压
* 电流积分（电荷）
* 卡尔文滤波
* 压力

## 神经网络是什么
神经网络是一种机器学习方法，其灵感来自人脑中神经元间信号传递的方式。神经网络特别适用于非线性关系建模，通常用于执行模式识别，以及对语音、视觉和控制系统中的目标或信号进行分类。

神经网络，特别是深度神经网络，以其在复杂识别应用中的出色表现而闻名，例如人脸识别、文本翻译和语音识别等应用。此类方法是推动高级驾驶辅助系统及其任务（包括车道分类和交通标志识别）创新的关键技术。

## 我们对于平台对于 SOC的用途，预估充电结束时间

首先不管是否使用神经网络 还是简单的程序 计算我们都 需要足够多的原始数据。

影响SOC速度指标的主要 外在因素就是电池 的循环次数，充电的环境温度，充电 的电压电流

1. 循环次数 我们可以 按批次+型号来锚定
2. 环境温度我们可以 以5摄氏度为单位进行采样
3. 充电的电压和电流咱们换电柜子相对稳定

先看下我们已有数据

~~~ sql
create table cos_power_report_db.battery_runtime_tb
(
    battery_pid                   String comment '电池出厂编号',
    receive_time                  DateTime comment '接收时间',
    total_voltage                 Decimal(5, 2) comment '电池总电压',
    current                       Decimal(4, 1) comment '电池电流',
    status                        Int8 comment '电池状态：0x01:静置；0x02:放电；0x03:充电；0x04加热并充电；0x05：只加热未充电',
    voltage_count                 Int8 comment '电芯电压数量（1-32）',
    voltages                      String comment '电芯电压列表，数组',
    temperature_count             Int8 comment '电池温度数量（1-10）',
    temperatures                  String comment '电池温度列表，数组',
    heat_film_temperature_count   Int8 comment '加热膜温度数（0-2）',
    heat_film_temperatures        String comment '加热膜温度列表，数组',
    environment_temperature_count Int8 comment '环境温度数（0-1）',
    environment_temperatures      String comment '环境温度列表，数组',
    mos_temperature_count         Int8 comment 'MOS温度数（0-2）',
    mos_temperatures              String comment 'MOS温度列表，数组',
    highest_voltage               Decimal(6, 3) comment '电池组最高单体电压',
    lowest_voltage                Decimal(6, 3) comment '电池组最低单体电压',
    average_voltage               Decimal(6, 3) comment '电池组平均单体电压',
    voltage_diff                  Decimal(6, 3) comment '压差',
    highest_temperature           Int16 comment '电池组最高电池温度',
    lowest_temperature            Int16 comment '电池组最低电池温度',
    total_capacity                Int32 comment '累计放电容量，预留',
    rated_capacity                Int16 comment '额定容量，预留',
    remain_capacity               Int16 comment '剩余容量，预留',
    soc                           Int8 comment '剩余容量',
    charge_mos_status             Int8 comment '充电MOS状态   1-闭合 0-断开 ',
    dis_charge_mos_status         Int8 comment '放电MOS状态  1-闭合 0-断开 ',
    heat_mos_status               Int8 comment '加热MOS状态   1-闭合 0-断开',
    lock_status                   Int8 comment '锁状态，1 开锁，0 锁定',
    circulate_count               Int32 comment '循环次数',
    highest_voltage_no            Int8 comment '最高电压编号',
    lowest_voltage_no             Int8 comment '最低电压编号',
    lowest_temperature_no         Int8 comment '最低温度编号',
    highest_temperature_no        Int8 comment '最低温度编号',
    create_time                   DateTime64(3) default now64(3) comment '创建时间'
)
~~~

原始数据采样方案，影响最小的方案是使用现有上报数据进行查找，但不一定有满足需求的数据，并且采样数据也涉及到更新。所以我们建议采用主动 采样
1. 指定要采样的型号和批次
2. 捕获电池 上报数据，当发现待采样电池，且当前 soc小于10%时，当前温度也没有采样数据。【温度使用仓温不要使用电池上报温度】
3. 锁定该仓不让进行换电，直到电池soc充到100%。

通过以上方案得到以下采样结果数据表：

| 型号批次，主要考虑不同容量或不同工艺问题 | 运营时长，半年为单位 | 温度，间隔5摄氏度 | soc值，间隔1% | 充满耗时 |
| ---------------------------------------- | -------------------- | ----------------- | ------------- | -------- |
| a001                                     | 0                    | 25                | 50%           | 120min   |
| a001                                     | 0                    | 25                | 60%           | 100min   |


## 神经网络计算过程

以上准备的原始数据做为训练集数据，首先进行归一化
然后构建神经中间层和输出层。然后通过 二次代价 函数和梯度下降法进行训练得到结果 模型

## SOH预测

锂离子电池健康状态（SOH）描述了电池当前老化程度，其估算难点在于缺乏明确统一的定义、无法直接测量以及难以确定数量合适、相关性高的估算输入量。为了克服上述问题，从容量的角度定义SOH，并将锂离子电池恒流-恒压充电过程中的电压、电流、温度曲线作为输入，提出采用一维深度卷积神经网络（CNN）实现锂离子电池容量估算以获取SOH

![NASA锂离子电池随机使用数据集中锂离子电池电压、电流、温度随SOH变化曲线](https://dgjsxb.ces-transaction.com/fileup/HTML/images/9d01a8e7540287b2750b164660398349.png)
NASA锂离子电池随机使用数据集中锂离子电池电压、电流、温度随SOH变化曲线


原始数据格式

~~~ shell
.
├── MatlabSamplePlots.m
├── README_RW_ChargeDischarge_RT.Rmd
├── README_RW_ChargeDischarge_RT.html
└── data
    ├── Matlab
    │   ├── RW10.mat
    │   ├── RW11.mat
    │   ├── RW12.mat
    │   └── RW9.mat
    └── R
        ├── RW10.Rda
        ├── RW11.Rda
        ├── RW12.Rda
        └── RW9.Rda
~~~

## 参考文献

* [基于卷积神经网络的锂离子电池SOH估算
](https://dgjsxb.ces-transaction.com/fileup/HTML/2020-19-4106.htm)
* [什么是循环神经网络 (RNN)](https://ww2.mathworks.cn/discovery/rnn.html)
* [循环神经网络（RNN）](https://www.tensorflow.org/tutorials/text/text_generation?hl=zh-cn)
* [时间序列预测](https://www.tensorflow.org/tutorials/structured_data/time_series?hl=zh_cn)
* []()