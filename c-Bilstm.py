import numpy as np
import pandas as pd
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from decimal import *
from tensorflow.keras import regularizers

TRAIN_MODE = 0
VALIDATION_MODE = 1
PREDICT_MODE = 2


def sample_split(parameter, is_scale=False):  # data processing

    df, resistor, point, start, stop, mode, __ = parameter
    dataset = df.iloc[:, 1:].values
    x = df.iloc[:, 0].values
    data_idx = list(range(start, stop))
    print(len(data_idx))
    print(point, start, stop)
    if mode == TRAIN_MODE:
        dataset = dataset[data_idx]
    if mode == VALIDATION_MODE:
        validation_idx = list([point])
        train_idx = list(set(data_idx) - set(validation_idx))
        train_set = dataset[train_idx]
        validation_set = dataset[validation_idx]
        dataset = np.vstack((train_set, validation_set))
    elif mode == PREDICT_MODE:
        train_set = dataset[data_idx]
        # 预测值的温度用dataset中的温度代替
        predict_list = [dataset[start, 0], resistor]
        dataset = np.vstack((train_set, predict_list))

    # 通过切比雪夫多项式增加特征项
    order = 10

    Rmax = np.max(dataset[:, 1])
    Rmin = np.min(dataset[:, 1])

    A = 2 / (math.log(Rmax) - math.log(Rmin))
    B = 1 - 2 * math.log(Rmax) / (math.log(Rmax) - math.log(Rmin))
    temp = np.empty((dataset.shape[0], order + 2))
    temp[:, 0] = dataset[:, 0]
    for i in range(dataset.shape[0]):
        for j in range(order + 1):
            temp[i, j + 1] = math.cos(j * math.acos(round(A * math.log(dataset[i, 1]) + B, 14)))
    dataset = temp

    # 归一化
    if is_scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        item = scaler.fit_transform(dataset)
        item[:, 1] = 1
        if mode == TRAIN_MODE:
            train_set = item
        elif mode == VALIDATION_MODE:
            train_set, validation_set = item[0:len(train_idx)], item[len(train_idx):]
        elif mode == PREDICT_MODE:
            train_set, predict_list = item[0:len(data_idx)], item[len(data_idx):]

    train_x, train_y = train_set[:, 1:dataset.shape[1]], train_set[:, 0]

    if mode == TRAIN_MODE:
        resistor = None
        validation_x = None
        validation_y = None
    elif mode == VALIDATION_MODE:
        validation_x, validation_y = validation_set[:, 1:dataset.shape[1]], validation_set[:, 0]
        resistor = None
    elif mode == PREDICT_MODE:
        resistor = predict_list[:, 1:dataset.shape[1]]
        validation_x = None
        validation_y = None

    return dataset, train_x, train_y, validation_x, validation_y, x, resistor


def bilstm_predict(parameter, hidden_number, count, epoch, load_weights=0, is_save_weights=True):
    dataset, train_x, train_y, validation_x, validation_y, x, resistor = sample_split(parameter, True)
    mode, flag = parameter[5:7]
    # 将数据转换成能够输入到bilstm模型里的结构
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    if mode == VALIDATION_MODE:
        validation_x = np.reshape(validation_x, (validation_x.shape[0], 1, validation_x.shape[1]))
    elif mode == PREDICT_MODE:
        resistor = np.reshape(resistor, (resistor.shape[0], 1, resistor.shape[1]))

    # 搭建深度学习模型
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_number, kernel_regularizer=regularizers.l1(0.00001)), input_shape=(1, train_x.shape[2])))

    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    if load_weights == 1:
        model.load_weights('my_model_weights.h5')
    if load_weights == 2:
        model.load_weights('my_model_weights %d %s.h5' % (count, flag))
    # 模型训练
    model.fit(train_x, train_y, epochs=epoch, batch_size=train_x.shape[0], verbose=2)
    # 预测
    train_predict = model.predict(train_x)
    if mode == VALIDATION_MODE:
        validation_predict = model.predict(validation_x)
    elif mode == PREDICT_MODE:
        predict = model.predict(resistor)

    model.save_weights('my_model_weights.h5')
    # 保存成六个模型
    if is_save_weights:
        model.save_weights('my_model_weights %d %s.h5' % (count, flag))

    # 反归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_y = scaler.fit_transform(dataset[:, 0].reshape(-1, 1))

    train_y = scaler.inverse_transform([train_y])
    train_predict = scaler.inverse_transform(train_predict)
    if mode == VALIDATION_MODE:
        validation_y, validation_predict = scaler.inverse_transform([validation_y]), scaler.inverse_transform(
            validation_predict)
    elif mode == PREDICT_MODE:
        predict = scaler.inverse_transform(predict)

    train_y = train_y.flatten()
    train_predict = train_predict.flatten()

    # 求MSE
    rmse = math.sqrt(mean_squared_error(train_y, train_predict))
    error = []
    for i in range(len(train_y)):
        error.append(train_y[i] - train_predict[i])

    print("RMSE = ", rmse)  # 均方误差MSE

    # 打印测试或预测点结果
    if mode == TRAIN_MODE:
        validation_y = None
        validation_predict = None
        predict = None
    if mode == VALIDATION_MODE:
        validation_y = validation_y.flatten()
        validation_predict = validation_predict.flatten()
        predict = None
    elif mode == PREDICT_MODE:
        predict = predict.flatten()
        validation_y = None
        validation_predict = None

    # 打印校准点的结果
    train_y_list = list(train_y)
    train_predict_list = list(train_predict)

    return rmse, validation_predict, validation_y, predict, train_predict_list, train_y_list


def auto_train(df, resistor, point, start, stop, mode, flag, hidden_number=6):
    # 将函数参数打包
    parameter = df, resistor, point, start, stop, mode, flag
    if mode == TRAIN_MODE or mode == VALIDATION_MODE:
        # 对模型预训练
        bilstm_predict(parameter, hidden_number, count=None, epoch=50000, load_weights=0, is_save_weights=False)
        # 训练三次取最小的均方差
        rmse_list = []
        for i in range(3):
            rmse, __, __, __, __, __ = bilstm_predict(parameter, hidden_number, count=None, epoch=500,
                                                    load_weights=1, is_save_weights=False)
            rmse_list.append(rmse)
        rmse_min = min(rmse_list)
        print("rmse_min", rmse_min)

    # 预测六次取平均
    count = 0
    result = []
    train_predict = []
    while count < 6:
        if mode == TRAIN_MODE:
            rmse, validation_predict, validation_y, predict, train_predict_list, train_y_list = bilstm_predict(parameter,
                                                                                                             hidden_number,
                                                                                                             count,
                                                                                                             epoch=1000,
                                                                                                             load_weights=1,
                                                                                                             is_save_weights=True)
            if rmse <= rmse_min:
                train_predict.append(train_predict_list)
                count += 1
        elif mode == VALIDATION_MODE:
            rmse, validation_predict, validation_y, predict, train_predict_list, train_y_list = bilstm_predict(parameter,
                                                                                                             hidden_number,
                                                                                                             count,
                                                                                                             epoch=1000,
                                                                                                             load_weights=1,
                                                                                                             is_save_weights=False)
            if rmse <= rmse_min:
                result.append(validation_predict[0])
                train_predict.append(train_predict_list)
                count += 1
        else:
            rmse, validation_predict, validation_y, predict, train_predict_list, train_y_list = bilstm_predict(parameter,
                                                                                                             hidden_number,
                                                                                                             count,
                                                                                                             epoch=0,
                                                                                                             load_weights=2,
                                                                                                             is_save_weights=False)
            result.append(predict[0])
            train_predict.append(train_predict_list)
            count += 1

    # 给校准点的拟合值做平均
    train_predict_mean = list(range(len(train_predict_list)))
    for i in range(len(train_predict_list)):
        train_predict_mean[i] = np.mean((train_predict[0][i], train_predict[1][i], train_predict[2][i],
                                         train_predict[3][i], train_predict[4][i], train_predict[5][i]))
    # 校准点误差
    error_train = list(map(lambda x: x[0] - x[1], zip(train_y_list, train_predict_mean)))

    if mode == TRAIN_MODE:
        res = None
        error_validation = None
    elif mode == VALIDATION_MODE:
        # 对预测结果作平均
        res = np.mean(result)
        # 获取验证集预测误差
        error_validation = validation_y - res

    elif mode == PREDICT_MODE:
        res = np.mean(result)
        error_validation = None

    return res, train_predict_mean, train_y_list, error_train, error_validation


def train(df, slice_low=0, slice_high=29, border_0=15, border_1=19, is_slice=False):
    mode = TRAIN_MODE
    np.random.seed(7)
    dataset = df.iloc[:, 1:].values
    if is_slice:
        __, train_predict_mean, train_y_list, error_train, __ = auto_train(df, resistor=None,
                                                                           point=None,
                                                                           start=slice_low,
                                                                           stop=slice_high + 1,
                                                                           mode=mode,
                                                                           flag='single')
        # 输出及打印校准点结果
        result_train = {"train_true": list(train_y_list), "train_predict": list(train_predict_mean),
                        "train_error": list(error_train)}
        res_train = pd.DataFrame(result_train)
        print(res_train)
        res_train.to_csv('result.txt', sep=',', index=False)
    else:
        # 第一分段训练
        __, train_predict_mean_0, train_y_list_0, error_train_0, __ = auto_train(df, resistor=None,
                                                                                 point=None,
                                                                                 start=0,
                                                                                 stop=border_1 + 1,
                                                                                 mode=mode,
                                                                                 flag='low')
        # 输出及打印校准点结果
        result_train_0 = {"train_true": list(train_y_list_0), "train_predict": list(train_predict_mean_0),
                          "train_error": list(error_train_0)}
        res_train_0 = pd.DataFrame(result_train_0)
        print(res_train_0)
        res_train_0.to_csv('result_0.txt', sep=',', index=False)
        # 第二分段训练
        __, train_predict_mean_1, train_y_list_1, error_train_1, __ = auto_train(df, resistor=None,
                                                                                 point=None,
                                                                                 start=border_0,
                                                                                 stop=dataset.shape[
                                                                                     0], mode=mode,
                                                                                 flag='high')
        # 输出及打印校准点结果
        result_train_1 = {"train_true": list(train_y_list_1), "train_predict": list(train_predict_mean_1),
                          "train_error": list(error_train_1)}
        res_train_1 = pd.DataFrame(result_train_1)
        print(res_train_1)
        res_train_1.to_csv('result_1.txt', sep=',', index=False)
    return 0


def predict(df, slice_low, slice_high, resistor, border_0=15, border_1=19, is_slice=False):
    np.random.seed(7)
    dataset = df.iloc[:, 1:].values
    point = None
    mode = PREDICT_MODE
    if is_slice:
        res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor,
                                                                                          point=None,
                                                                                          start=slice_low,
                                                                                          stop=slice_high + 1,
                                                                                          mode=mode,
                                                                                          flag='single')
        print("res", res)
        res_0 = '无'
        res_1 = '无'

    else:
        # 预测点在第一分段
        if resistor >= dataset[border_0, 1]:

            res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor,
                                                                                              point=None,
                                                                                              start=0,
                                                                                              stop=border_1 + 1,
                                                                                              mode=mode,
                                                                                              flag='low')

            print("res", res)
            res_0 = '无'
            res_1 = '无'

        # 预测点在重叠区域
        elif resistor > dataset[border_1, 1]:
            res_0, train_predict_mean_0, train_y_list_0, error_train_0, error_validation_0 = auto_train(df, resistor,
                                                                                                        point=None,
                                                                                                        start=0,
                                                                                                        stop=border_1 + 1,
                                                                                                        mode=mode,
                                                                                                        flag='low')
            res_1, train_predict_mean_1, train_y_list_1, error_train_1, error_validation_1 = auto_train(df, resistor,
                                                                                                        point=None,
                                                                                                        start=border_0,
                                                                                                        stop=
                                                                                                        dataset.shape[
                                                                                                            0],
                                                                                                        mode=mode,
                                                                                                        flag='high')
            # 重叠区间优化
            midpoint = (border_0 + border_1) / 2
            midpoint = int(midpoint)
            if resistor >= dataset[midpoint, 1]:
                res = res_0
            else:
                res = res_1
            # 输出预测结果
            print("res_0", res_0)
            print("res_1", res_1)
            print("res", res)

        # 预测点在第二分段
        else:
            res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor, point=None,
                                                                                              start=border_0,
                                                                                              stop=dataset.shape[0],
                                                                                              mode=mode,
                                                                                              flag='high')

            print("res", res)
            res_0 = '无'
            res_1 = '无'

    return res, res_0, res_1


def validating(df, slice_low, slice_high, point, border_0=15, border_1=19, is_slice=False):
    np.random.seed(7)
    dataset = df.iloc[:, 1:].values
    resistor = None
    mode = VALIDATION_MODE
    if is_slice:
        res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor,
                                                                                          point,
                                                                                          start=slice_low,
                                                                                          stop=slice_high + 1,
                                                                                          mode=mode,
                                                                                          flag=None)

        print("res_validation,error_validation", res, error_validation)
        result_train = {"train_true": list(train_y_list), "train_predict": list(train_predict_mean),
                        "train_error": list(error_train)}
        # 打印结果输出文档
        res_train = pd.DataFrame(result_train)
        print(res_train)
        res_train.to_csv('result.txt', sep=',', index=False)
        with open('result.txt', mode='a')as f:
            f.write('res_test %.15f\n error_test %.15f\n' % (res, error_validation))
    else:
        # 预测点在第一分段
        if point <= border_0:

            res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor,
                                                                                              point,
                                                                                              start=0,
                                                                                              stop=border_1 + 1,
                                                                                              mode=mode,
                                                                                              flag=None)

            print("res_validation,error_validation", res, error_validation)
            result_train = {"train_true": list(train_y_list), "train_predict": list(train_predict_mean),
                            "train_error": list(error_train)}

            res_train = pd.DataFrame(result_train)
            print(res_train)
            res_train.to_csv('result.txt', sep=',', index=False)
            with open('result.txt', mode='a')as f:
                f.write('res_test %.15f\n error_test %.15f\n' % (res, error_validation))
        # 预测点在重叠区域
        elif point < border_1:
            res_0, train_predict_mean_0, train_y_list_0, error_train_0, error_validation_0 = auto_train(df, resistor,
                                                                                                        point,
                                                                                                        start=0,
                                                                                                        stop=border_1 + 1,
                                                                                                        mode=mode,
                                                                                                        flag=None)
            res_1, train_predict_mean_1, train_y_list_1, error_train_1, error_validation_1 = auto_train(df, resistor,
                                                                                                        point,
                                                                                                        start=border_0,
                                                                                                        stop=
                                                                                                        dataset.shape[
                                                                                                            0],
                                                                                                        mode=mode,
                                                                                                        flag=None)
            # 重叠区间优化
            midpoint = (border_0 + border_1) / 2

            if point <= midpoint:
                res = res_0
            elif point == midpoint:
                res = (res_0 + res_1) / 2
            else:
                res = res_1

            error_validation = (error_validation_0 + error_validation_1) / 2
            # 输出预测结果
            print("res_validation_0,error_validation_0", res_0, error_validation_0)
            print("res_validation_1,error_validation_1", res_1, error_validation_1)
            print("res_validation,error_validation", res, error_validation)
            # 打印校准点结果
            result_train_0 = {"train_true": list(train_y_list_0), "train_predict": list(train_predict_mean_0),
                              "train_error": list(error_train_0)}
            res_train_0 = pd.DataFrame(result_train_0)
            print(res_train_0)
            res_train_0.to_csv('result_0.txt', sep=',', index=False)
            with open('result_0.txt', mode='a')as f:
                f.write('res_test_0 %.15f\n error_test_0 %.15f\n res_test %.15f\n error_test '
                        '%.15f\n' % (res_0, error_validation_0, res, error_validation))

            result_train_1 = {"train_true": list(train_y_list_1), "train_predict": list(train_predict_mean_1),
                              "train_error": list(error_train_1)}
            res_train_1 = pd.DataFrame(result_train_1)
            print(res_train_1)
            res_train_1.to_csv('result_1.txt', sep=',', index=False)
            with open('result_1.txt', mode='a')as f:
                f.write('res_test_1 %.15f\n error_test_1 %.15f\n res_test %.15f\n error_test '
                        '%.15f\n' % (res_1, error_validation_1, res, error_validation))
        # 预测点在第二分段
        else:
            res, train_predict_mean, train_y_list, error_train, error_validation = auto_train(df, resistor, point,
                                                                                              start=border_0,
                                                                                              stop=dataset.shape[0],
                                                                                              mode=mode,
                                                                                              flag=None)
            print("res_validation,error_validation", res, error_validation)

            result_train = {"train_true": list(train_y_list), "train_predict": list(train_predict_mean),
                            "train_error": list(error_train)}
            res_train = pd.DataFrame(result_train)
            print(res_train)
            res_train.to_csv('result.txt', sep=',', index=False)
            with open('result.txt', mode='a')as f:
                f.write('res_test %.15f\n error_test %.15f\n' % (res, error_validation))
    return 0


if __name__ == '__main__':
    train()

