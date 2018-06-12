# coding=utf-8

import xlrd
from xlrd import xldate_as_datetime
import numpy as np
import csv


from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit

def read_xls(xls_file):
    data = xlrd.open_workbook(xls_file)
    table = data.sheet_by_index(0)
    nrows = table.nrows
    data_li = []
    for i in range(nrows):
        data_li.append(table.row_values(i))
        # print str(xldate_as_datetime(table.row_values(i)[0],0))
    # print data_li.__len__()

    return data_li

def get_y():
    data = read_xls('data/y.xlsx')
    y = []
    for row in data:
        price = str(row[1])
        if len(price) > 1:
            if int(row[0]) > 20170630:
                continue
            day = str(int(row[0]))
            value = row[1]
            y.append([day, value])
    print y
    return y

def get_x1():
    data = read_xls('data/x1.xls')
    x1 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00','').replace('-','')
        value = row[1]
        x1.append([day,value])

    print x1
    return x1

def get_x2():
    data = read_xls('data/x2.xls')
    x2 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00','').replace('-','')
        value = row[1]
        x2.append([day, value])

    print x2
    return x2

def get_x3():
    data = read_xls('data/x3.xls')
    x3 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = row[1]
        x3.append([day, value])

    print x3
    return x3

def get_x4():
    data = read_xls('data/x4.xlsx')
    price_dict = {}
    for row in data:
        day = row[0]
        type = row[1]
        value = row[2]
        if day not in price_dict:
            price_dict[day] = [value]
        else:
            price_dict[day].append(value)

    # 求平均
    avg_dict = {}
    for day in price_dict:
        values = np.array(price_dict[day])
        avg_dict[day] = np.mean(values)

    x4 = []
    for row in data:
        day = row[0]
        value = avg_dict[day]
        day = str(xldate_as_datetime(day, 0)).replace(' 00:00:00', '').replace('-', '')
        x4.append([day, value])
    return x4

def get_x5():
    data = read_xls('data/x5.xls')
    x5 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = row[1]
        x5.append([day, value])

    print x5
    return x5

def get_x6():
    data = read_xls('data/x6.xlsx')
    x6 = []
    for row in data:
        day = str(row[0]).replace('-', '')
        # day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = float(row[1].replace(',',''))
        x6.append([day, value])

    print x6
    return x6

def get_x7():
    data = read_xls('data/x7.xlsx')
    x7 = []
    for row in data:
        day = str(int(row[0]))
        value = float(row[1])
        x7.append([day, value])

    print x7
    return x7

def get_new_x7():
    data = read_xls('data/new_x7.xlsx')
    x = []
    for row in data:
        day = str(int(row[0]))[0:6]
        x.append([day, row[1]])
    print x
    return x

def get_x8():
    data = read_xls('data/x8.xlsx')
    x8 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = float(row[1])
        x8.append([day, value])

    print x8
    return x8

def get_x9():
    data = read_xls('data/x9.xls')
    x9 = []
    for row in data:
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = float(row[1])
        x9.append([day, value])

    print x9
    return x9

def get_x10():
    data = read_xls('data/x10.xlsx')
    x10 = []
    for row in data:
        day = str(int(row[0]))
        if u'低' in row[1]:
            value = 1
        elif u'中' in row[1]:
            value = 2
        elif u'高' in row[1]:
            value = 3
        else:
            value = 0
        x10.append([day, value])

    print x10
    return x10

def get_x11():
    data = read_xls('data/x11.xlsx')
    x11 = []
    for row in data:
        # print row
        day = str(xldate_as_datetime(row[0], 0)).replace(' 00:00:00', '').replace('-', '')
        value = row[1]
        x11.append([day, value])

    print x11
    return x11

def get_x1_index(day, x1):
    for row in x1:
        if day == row[0]:
            x1i = row[1]
            return x1i
    return 0

def get_x2_index(day, x2):
    day = int(day)
    for row in x2:
        if day < int(row[0]):
            x2i = row[1]
            return x2i
    return 0

def get_x3_index(day, x3):

    x3i = 0
    # 先取是否相等
    for row in x3:
        if day == row[0]:
            x3i = row[1]

    if x3i > 0:
        return x3i

    # 取中间数
    day = int(day)
    for j in range(len(x3)-1):
        if day > int(x3[j][0]) and day < int(x3[j+1][0]):
            x3i = (float(x3[j][1] + float(x3[j+1][1]))) / 2
            return x3i
    return x3i

def get_x4_index(day, x4):

    x4i = 0
    for row in x4:
        if day == row[0]:
            x4i = row[1]
            return x4i
    return x4i

def get_x5_index(day, x5):

    x5i = 0
    day = day[0:6]
    for row in x5:
        row_day = row[0][0:6]
        if day == row_day:
            x5i = row[1]
    #     if
    #     if day == row[0]:
    #         x4i = row[1]
    #         return x4i
    return x5i

def get_x6_index(day, x6):

    x6i = 0
    for row in x6:
        row_day = row[0]
        if day == row_day:
            x6i = row[1]
    return x6i

def get_x7_index(day, x7):

    x7i = 0
    day = day[0:4]
    for row in x7:
        row_day = str(int(row[0]))
        if day == row_day:
            x7i = row[1]
    return x7i

def get_new_x7_index(day, new_x7):

    x7i = 0
    day = int(day[0:6])
    for row in new_x7:
        row_day = int(row[0])
        # print day, row_day
        if day >= row_day:
            x7i = row[1]
            # break
    return x7i


def get_x8_index(day, x8):

    x8i = 0
    day = int(day[0:6])
    for row in x8:
        row_day = int(row[0][0:6])
        # print day, row_day
        if day >= row_day:
            x8i = row[1]
            # break
    return x8i

def get_x9_index(day, x9):

    x9i = 0
    for row in x9:
        row_day = row[0]
        if day == row_day:
            x9i = row[1]
    return x9i

def get_x10_index(day, x10):

    x10i = 0
    for row in x10:
        row_day = row[0]
        if day == row_day:
            x10i = row[1]
    return x10i

def get_x11_index(day, x11):

    x11i = 0
    day = int(day[0:6])
    for row in x11:
        row_day = int(row[0][0:6])
        # print day, row_day
        if day == row_day:
            x11i = row[1]
            # break
    return x11i


def create_dataset():
    y = get_y()
    x1 =get_x1()
    x2 =get_x2()  # 按周算
    x3 = get_x3()
    x4 = get_x4()
    x5 = get_x5()
    x6 = get_x6()
    # x7 = get_x7()
    x7 = get_new_x7()
    x8 = get_x8()
    x9 = get_x9()
    x10 = get_x10()
    x11 = get_x11()

    data_set = []
    # print y
    for row in y:
        day = row[0]
        yi = row[1]
        x1i = get_x1_index(day, x1)
        # print day, yi, x1i
        x2i = get_x2_index(day, x2)
        # print day, yi, x2i
        x3i = get_x3_index(day, x3)
        # print day, yi, x3i
        x4i = get_x4_index(day, x4)
        # print day, yi, x4i
        x5i = get_x5_index(day, x5)
        # print day, yi, x5i
        x6i = get_x6_index(day, x6)
        # print day, yi, x6i
        # x7i = get_x7_index(day, x7)
        x7i = get_new_x7_index(day, x7)
        # print day, yi, x7i
        x8i = get_x8_index(day, x8)
        # print day, yi, x8i
        x9i = get_x9_index(day, x9)
        # print day, yi, x9i
        x10i = get_x10_index(day, x10)
        # print day, yi, x10i

        x11i = get_x11_index(day, x11)
        # print day, yi, x11i

        one = [day, yi,x1i,x2i,x3i,x4i,x5i,x6i,x7i,x8i,x9i,x10i,x11i]
        data_set.append(one)

    data_file = 'data/new_data.csv'
    out = open(data_file, 'w')
    csv_writer = csv.writer(out)
    header = ['date', 'y','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']
    csv_writer.writerow(header)
    for row in data_set:
        csv_writer.writerow(row)

# create_dataset()

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))


def read_dataset():
    data_file = 'data/new_data_4.csv'
    csv_reader = csv.reader(open(data_file))
    X = []
    Y = []
    head = True
    for row in csv_reader:
        if head == True:
            head = False
            continue
        y = float(row[1])
        if y == 0:
            continue
        x1 = float(row[2])
        x2 = float(row[3])
        x3 = float(row[4])
        x4 = float(row[5])
        x5 = float(row[6])
        x6 = float(row[7])
        x7 = float(row[8])
        x8 = float(row[9])
        x9 = float(row[10])
        x10 = float(row[11])
        x11 = float(row[12])
        X.append([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    # sigmoid 归一化
    # for i in range(11):
    #     X[:, i] = X[:, i] / max(X[:, i])
    #     X[:, i] = sigmoid(X[:, i])
    # # print X
    # Y = Y / max(Y)
    # Y = sigmoid(Y)
    # print Y
    # new_data = []
    # for i in range(0,len(X), 1):
    #     # print Y[i]
    #     # print X[i]
    #     # print list(Y_minmax[i])
    #     data = [Y[i]] + list(X[i])
    #     print data
    #     new_data.append(data)

    # # 最大最小化
    Y = np.reshape(Y, [-1, 1])
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    Y_minmax = min_max_scaler.fit_transform(Y)
    print Y_minmax

    for i in range(11):
        if i == 2 or i == 3 or i == 7 or i == 8:
            X_minmax[:, i] = sigmoid(X_minmax[:, i])


    new_data = []
    for i in range(0,len(X_minmax), 1):
        print list(Y_minmax[i])
        data = list(Y_minmax[i]* 0.998 +0.001) + list(X_minmax[i])
        print data
        new_data.append(data)
    #



    data_file = 'data/norm_new_data_2016.csv'
    out = open(data_file, 'w')
    csv_writer = csv.writer(out)
    header = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
    csv_writer.writerow(header)
    for row in new_data:
        csv_writer.writerow(row)
    #
    # # Y_minmax = np.concatenate(Y_minmax)
    #
    # ss = ShuffleSplit(n_splits=1, test_size=0.1,random_state=0)
    # for train_index, test_index in ss.split(X):
    #     X_train = X_minmax[train_index]
    #     y_train = Y_minmax[train_index]
    #     X_dev = X_minmax[test_index]
    #     y_dev = Y_minmax[test_index]


    # return X_train, y_train, X_dev, y_dev


    # print X_minmax
    # print Y_minmax




read_dataset()

def read():
    data_file = 'data/data2.csv'
    csv_reader = csv.reader(open(data_file))
    X = []
    Y = []
    for row in csv_reader:
        if int(row[0]) > 20150630:
            continue
        # print row[0]
        y = float(row[1])
        # x1 = float(row[2])
        # x2 = float(row[3])
        # x3 = float(row[4])
        # x4 = float(row[5])
        # x5 = float(row[6])
        # x6 = float(row[7])
        # x7 = float(row[8])
        x8 = float(row[9])
        # x10 = float(row[11])
        # x11 = float(row[12])
        X.append(x8)
        Y.append(y)
        print row[0], round(y,3), x8
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

import math
def pearon(vector1, vector2):
    n = len(vector1)
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))

    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    num = p_sum - (1.0 * sum1 * sum2 / n)
    den = math.sqrt((sum1_pow-pow(sum1,2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

# X, Y = read()
# print X.shape
# print pearon(X, Y)