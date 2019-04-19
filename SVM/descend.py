import numpy as np
import csv
import sys
import pprint
import random
import time


def read_file(filename):
    lines = open(filename).readlines()
    data_set = []
    label_set = []
    for i in range(0, len(lines)):
        data = lines[i].split(" ")
        data_set.append([])
        for j in range(0, len(data) - 1):
            data_set[i].append(float(data[j]))
        label_set.append(float(data[-1]))
    return np.array(data_set), np.array(label_set)


class SVM():
    def __init__(self, x, y, epochs, start, learning_rate=0.01):
        self.x = np.c_[np.ones((x.shape[0])), x]
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.uniform(size=np.shape(self.x)[1],)
        self.start = start
        self.number = 0

    def get_loss(self, x, y):
        loss = max(0, 1-y*np.dot(x, self.w))
        return loss

    def cal_sgd(self, x, y, w):
        if y*np.dot(x, w) < 1:
            w = w - self.learning_rate * (-y * x)
        else:
            w = w
        return w

    def train(self):
        for epoch in range(self.epochs):
            randomize = np.arange(len(self.x))
            np.random.shuffle(randomize)
            x = self.x[randomize]
            y = self.y[randomize]
            loss = 0
            for xi, yi in zip(x, y):
                loss += self.get_loss(xi, yi)
                self.w = self.cal_sgd(xi, yi, self.w)
            # print('epoch: {0} loss: {1}'.format(epoch, loss))
            if time.time() > (55+self.start):
                break
            if loss == 0:
                self.number += +1
                if self.number == 3:
                    break

    def predict(self, x):
        x_test = np.c_[np.ones((x.shape[0])),x]
        return np.sign(np.dot(x_test, self.w))


# L = []
# A = []
# D = []
# index = []
#
#
# class SMO:
#     def __init__(self, data, label, C, tolar):
#         self.data = data  # 数据集
#         self.label = label  # 标签集
#         self.C = C  # outliers用于控制权重，给定常量
#         self.tolar = tolar  # 能容忍的极限值
#         self.colomn = len(data)  # 个数
#         self.a = np.array(np.zeros(self.colomn), dtype='float64')  # 拉格朗日乘子
#         self.E = np.array(np.zeros((self.colomn, 2)))  # 记录预测值与真实值之差
#         self.b = 0.0  # 分类函数中b
#         self.ker = np.array(np.zeros((self.colomn, self.colomn), dtype='float64'))  # 记录核函数的值
#         for i in range(0, self.colomn):
#             for j in range(0, self.colomn):
#                 x = data[i]
#                 y = data[j]
#                 self.ker[i, j] = self.kernel(x, y)
#
#     def kernel(self, x, y):
#         # return np.exp(sum((x-y)*(x-y))/(-1*20*20*2))   # 高斯核
#         return sum(x * y)
#
#     def cal_value(self, k):
#         temp = np.dot(self.a*self.label, self.ker[:, k]) + self.b
#         return temp-float(self.label[k])
#
#     def update_value(self, k):
#         temp = self.cal_value(k)
#         self.E[k] = [1, temp]
#
#     def select_j(self, i, i_value):
#         max_value = 0.0
#         select_j = 0
#         j_value = 0.0
#         valid_list = np.nonzero(self.E[:, 0])[0]
#         if len(valid_list) > 1:
#             for k in valid_list:
#                 if k == i:
#                     continue
#                 k_value = self.cal_value(k)
#                 temp = abs(i_value - k_value)
#                 if temp > max_value:
#                     select_j = k
#                     max_value = temp
#                     j_value = k_value
#             return select_j, j_value
#         else:
#             select_j = int(random.uniform(0, self.colomn))
#             while select_j == i:
#                 select_j = int(random.uniform(0, self.colomn))
#             j_value = self.cal_value(select_j)
#             return select_j, j_value
#
#     def KKT(self, i):
#         i_value = self.cal_value(i)
#         if (self.label[i] * i_value < -self.tolar and self.a[i] < self.C) or \
#             (self.label[i] * i_value > self.tolar and self.a[i] > 0):
#             self.update_value(i)
#             j, j_value = self.select_j(i, i_value)
#             i_old = self.a[i]
#             j_old = self.a[j]
#             self.update_a(i, i_value, j, j_value)
#             self.update_value(j)
#             if abs(j_old-self.a[j]) < self.tolar:
#                 return 0
#             self.a[i] += self.label[i] * self.label[j] * (j_old - self.a[j])
#             self.update_value(i)
#             self.update_b(i_value, j_value, i, j, i_old, j_old)
#             return 1
#         return 0
#
#     def update_a(self, i, i_value, j, j_value):
#         if self.label[i] != self.label[j]:
#             L = max(0, self.a[j] - self.a[i])
#             H = min(self.C, self.C + self.a[j] - self.a[i])
#         else:
#             L = max(0, self.a[j] + self.a[i] - self.C)
#             H = min(self.C, self.a[j] + self.a[i])
#         e = self.ker[i, i] + self.ker[j, j] - 2 * self.ker[i, j]
#         if e <= 0:
#             return 0
#         self.a[j] += self.label[j] * (i_value - j_value) / e
#         if self.a[j] > H:
#             self.a[j] = H
#         elif self.a[j] < L:
#             self.a[j] = L
#
#     def update_b(self, i_value, j_value, i, j, i_old, j_old):
#         b1 = self.b - i_value - self.label[i] * (self.a[i] - i_old) * self.ker[i, i] - self.label[j] * \
#              (self.a[j] - j_old) * self.ker[i, j]
#         b2 = self.b - j_value - self.label[i] * (self.a[i] - i_old) * self.ker[i, j] - self.label[j] * \
#              (self.a[j] - j_old) * self.ker[j, j]
#         if 0 < self.a[i] < self.C:
#             self.b = b1
#         elif 0 < self.a[j] < self.C:
#             self.b = b2
#         else:
#             self.b = (b1 + b2) / 2
#
#     def smo(self):
#         it = 0
#         num_a_change = 0  # 乘子改变的次数
#         flag = True
#         while it < 10000 and ((num_a_change > 0) or flag):
#             num_a_change = 0
#             if flag:
#                 for i in range(0, self.colomn):
#                     num_a_change += self.KKT(i)
#                 it += 1
#             else:
#                 nobound = np.nonzero((self.a > 0) * (self.a < self.C))[0]
#                 for i in nobound:
#                     num_a_change += self.KKT(i)
#                 it += 1
#             if flag:
#                 flag = False
#             elif num_a_change == 0:
#                 flag = True
#         index = np.nonzero(self.a)[0]
#         L = self.label[index]
#         A = self.a[index]
#         D = self.data[index]
#
#     def predict(self, test):
#         result = []
#         m = np.shape(test)[0]
#         # print(m)
#         for i in range(0, m):
#             temp = self.b
#             for j in range(len(index)):
#                 temp += A[j] * L[j] * self.kernel(D[j], test[i,:])
#             while temp == 0:
#                 temp = random.uniform(-1, 1)
#             if temp > 0:
#                 temp = 1
#             else:
#                 temp = -1
#             result.append(temp)
#         return result


def load_test_data(filename):
    lines = open(filename).readlines()
    data_set = []
    for i in range(0, len(lines)):
        data = lines[i].split(" ")
        data_set.append([])
        for j in range(0, len(data)):
            data_set[i].append(float(data[j]))
    return np.array(data_set)


def main():
    start = time.time()
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    data, label = read_file(filename1)
    test_data = load_test_data(filename2)
    svm = SVM(data, label, 1000000, start)
    svm.train()
    # print(time.time() - start)
    # print(svm.number)
    test_result1 = svm.predict(test_data)
    # print(svm.index)

    # a = len(label)
    # b = 0
    for i in range(len(test_result1)):
        # if label[i] == test_result1[i]:
        #     b = b+1
        print(test_result1[i])
    # print(b/a)

    # svm = SMO(data, label, 200, 0.00001)
    # svm.smo()
    # test_result2 = svm.predict(data)
    # c = 0
    # for i in range(len(test_result2)):
    #     if label[i] == test_result2[i]:
    #         c = c + 1
    # print(c / a)


if __name__ == "__main__":
    main()
