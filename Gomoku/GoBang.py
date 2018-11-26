import numpy as np
import random
import math
import time
import copy
import operator

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
'''score = [(50000,(1,1,1,1,1)),(4320,(0,1,1,1,1,0)), #五，活四
         (720,(1,1,1,1,0)),(720,(0,1,1,1,1)), #眠四
         (720,(1,0,1,1,1)),(720,(1,1,0,1,1)), #眠四
         (720,(1,1,1,0,1)),(720,(0,1,1,1,0)), #眠四，活三
         (720,(0,1,1,0,1,0)),(720,(0,1,0,1,1,0)), #活三
         (120,(0,0,1,1,1)),(120,(0,0,1,1,1)), #眠三
         (120,(1,1,0,1,0)),(20,(0,0,1,1,0)), #眠三，活二
         (20,(0,1,1,0,0))]'''


class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # you are white or black
        self.color = color
        # the max time you should use
        self.time_out = time_out
        # add the decision into candidate_list. System will get the end of the candidate_list as decision
        self.candidate_list = []
        self.searchDeep = 4
        self.MAX = 1 << 28
        self.MIN = -self.MAX
        self.waiting_list = []
        self.start = -1
        self.old_chessboard = [[0 for x in range(chessboard_size)] for y in range(chessboard_size)]

    # the input is current chessboard
    def go(self, chessboard):
        # clear candidate_list
        self.candidate_list.clear()
        if self.start == -1 and self.empty_chessboard(chessboard) and self.color == -1:  # 说明自己先手,目前棋盘上无子
            result = (math.floor(self.chessboard_size / 2), math.floor(self.chessboard_size / 2))
            if self.waiting_list.__contains__(result):  # 将选定点从待选择表中删除
                self.waiting_list.remove(result)
            self.start = 1
            self.neighbor_point(result[0], result[1], chessboard)  # 将落子的邻居加入待选择表
            self.copy_chessboard(chessboard, result[0], result[1])
            self.candidate_list.append((result[0], result[1]))
            print(result[0], result[1])
            return
        else:  # 自己为白棋或双方已经下过一轮
            last_point = self.find_last_point(chessboard)  # 得到上一步对方落子位置
            if self.waiting_list.__contains__(last_point):  # 将选定点从待选择表中删除
                self.waiting_list.remove(last_point)
            self.neighbor_point(last_point[0],last_point[1], chessboard)  # 将落子的邻居加入待选择表
            node = Node()
            self.dfs(node, 0, self.MAX, self.MIN, chessboard)
            best_list = node.get_best()
            optimal = best_list[0].point
            if self.waiting_list.__contains__(optimal):  # 将选定点从待选择表中删除
                self.waiting_list.remove(optimal)
            self.neighbor_point(optimal[0], optimal[1], chessboard)  # 将落子的邻居加入待选择表
            self.copy_chessboard(chessboard, optimal[0], optimal[1])
            self.candidate_list.append((optimal[0], optimal[1]))
            print(optimal[0], optimal[1])

    def dfs(self, root, deep, alpha, beta, chessboard):
        if deep == self.searchDeep:  # 达到搜索深度，直接计算
            root.mark = self.mark(chessboard)
            return
        list1 = copy.deepcopy(self.waiting_list)
        i = -1
        while i < list1.__len__()-1:
            i = i+1
            obj = list1[i]
            point = (obj[0], obj[1])
            node = Node()
            node.set_point(point)
            root.add_child(node)
            flag = list1.__contains__(point)
            temp1 = deep & 1
            if temp1 == 1:
                chessboard[point[0]][point[1]] = -self.color
            else:
                chessboard[point[0]][point[1]] = self.color
            # 如果waiting_list中某点连成五个子，结束
            if self.end(chessboard, point[0], point[1]):
                root.set_best(node)
                root.mark = self.MAX*chessboard[point[0]][point[1]]
                chessboard[point[0]][point[1]] = 0
                return
            reminder = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    else:
                        x = point[0] + i
                        y = point[1] + j
                        if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size:
                            if chessboard[x][y] == 0:
                                p = (x, y)
                                if not self.waiting_list.__contains__(p):
                                    self.waiting_list.append(p)
                                    reminder.append(p)
            if flag:
                self.waiting_list.remove(point)
            self.dfs(root.get_last_child(), deep+1, alpha, beta, chessboard)
            chessboard[point[0]][point[1]] = 0
            if flag:
                self.waiting_list.append(point)
            for pos in reminder:
                self.waiting_list.remove(pos)
            # alpha beta 剪枝
            # min层
            table = deep & 1
            if table == 1:
                if root.get_best() == [] or root.get_last_child().mark < root.get_best()[0].mark:
                    root.set_best(root.get_last_child())
                    root.mark = root.get_best()[0].mark
                    if root.mark <= self.MIN:
                        root.mark = root.mark + deep
                    beta = min(root.mark, beta)
                if root.mark <= alpha:
                    return
            # max
            else:
                if root.get_best() == [] or root.get_last_child().mark > root.get_best()[0].mark:
                    root.set_best(root.get_last_child())
                    root.mark = root.get_best()[0].mark
                    if root.mark == self.MAX:
                        root.mark = -deep
                    alpha = max(root.mark, alpha)
                if root.mark >= beta:
                    return

    def empty_chessboard(self, chessboard):
        for i in range(0, self.chessboard_size):
            for j in range(0, self.chessboard_size):
                if chessboard[i][j] == 0:
                    continue
                else:
                    return False
        return True

    # 复制保存上一轮的棋盘
    def copy_chessboard(self, chessboard, x, y):
        self.old_chessboard = copy.deepcopy(chessboard)
        self.old_chessboard[x][y] = self.color

    # 得到两次棋盘上点不同的位置
    def find_last_point(self, chessboard):
        for i in range(0, self.chessboard_size):
            for j in range(0, self.chessboard_size):
                if chessboard[i][j] != self.old_chessboard[i][j]:
                    last_point = (i, j)
                    return last_point

    # 将落子的邻居加入待选表
    def neighbor_point(self, a, b, chessboard):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                else:
                    x = a + i
                    y = b + j
                    if 0 <= x < self.chessboard_size and 0 <= y < self.chessboard_size and chessboard[x][y]==0:
                        p = (x, y)
                        if not self.waiting_list.__contains__(p):
                            self.waiting_list.append(p)

    # 评分公式
    def mark(self, chessboard):
        score = 0
        base = 20
        for i in range(0, self.chessboard_size):
            for j in range(0, self.chessboard_size):
                if chessboard[i][j] != 0:
                    re_i = i
                    re_j = j
                    # 列：
                    initial = 5
                    count = 1
                    x = re_i
                    y = re_j - 1
                    flag_down_end = False
                    flag_up_end = False
                    while y >= 0 and chessboard[x][y] == chessboard[i][j]:
                        count = count+1
                        y = y-1
                    if y >= 0 and chessboard[x][y] == 0:
                        flag_down_end = True
                    x = re_i
                    y = re_j+1
                    while y < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
                        count = count+1
                        y = y+1
                    if y < self.chessboard_size and chessboard[x][y] == 0:
                        flag_up_end = True
                    if flag_up_end and flag_down_end:  # 说明是活棋
                        score = score+chessboard[i][j]*count*count
                        '''if count == 1:
                            score = score + chessboard[i][j] * base
                        elif count == 2:
                            score = score + chessboard[i][j] * initial * base
                        else:
                            n = count - 2
                            for k in range(0, n):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                    elif flag_down_end or flag_up_end:
                        score = score + chessboard[i][j] * count * count/4
                        '''if count == 2:
                            score = score + chessboard[i][j] * base
                        elif count == 3:
                            score = score + chessboard[i][j] * initial * base
                        elif count > 3:
                            n = count - 3
                            for k in range(0, n):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                    # 行：
                    initial = 5
                    count = 1
                    x = re_i - 1
                    y = re_j
                    flag_right_end = False
                    flag_left_end = False
                    while x >= 0 and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        x = x - 1
                    if x >= 0 and chessboard[x][y] == 0:
                        flag_left_end = True
                    x = re_i + 1
                    y = re_j
                    while x < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        x = x + 1
                    if x < self.chessboard_size and chessboard[x][y] == 0:
                        flag_right_end = True
                    if flag_right_end and flag_left_end:  # 说明是活棋
                        score = score + chessboard[i][j] * count * count
                        '''if count == 1:
                            score = score + chessboard[i][j] * base
                        elif count == 2:
                            score = score + chessboard[i][j] * initial * base
                        else:
                            n = count - 2
                            for k in range(0, n):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                    elif flag_right_end or flag_left_end:
                        score = score + chessboard[i][j] * count * count/4
                        '''if count == 2:
                            score = score + chessboard[i][j] * base
                        elif count == 3:
                            score = score + chessboard[i][j] * initial * base
                        elif count > 3:
                            n = count - 3
                            for k in range(0, n):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                    # 左对角线
                    initial = 5
                    count = 1
                    x = re_i - 1
                    y = re_j - 1
                    flag_down_end = False
                    flag_up_end = False
                    while y >= 0 and x >= 0 and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        y = y - 1
                        x = x - 1
                    if y >= 0 and x >= 0 and chessboard[x][y] == 0:
                        flag_down_end = True
                    x = re_i + 1
                    y = re_j + 1
                    while x < self.chessboard_size and y < self.chessboard_size \
                            and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        y = y + 1
                        x = x + 1
                    if x < self.chessboard_size and y < self.chessboard_size and chessboard[x][y] == 0:
                        flag_up_end = True
                    if flag_up_end and flag_down_end:  # 说明是活棋
                        '''if count == 1:
                            score = score + chessboard[i][j] * base
                        elif count == 2:
                            score = score + chessboard[i][j] * initial * base
                        else:
                            for k in range(0, count - 2):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                        score = score + chessboard[i][j] * count * count
                    elif flag_down_end or flag_up_end:
                        '''if count == 2:
                            score = score + chessboard[i][j] * base
                        elif count == 3:
                            score = score + chessboard[i][j] * initial * base
                        elif count > 3:
                            for k in range(0, count - 3):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                        score = score + chessboard[i][j] * count * count/4

                    # 右对角线
                    initial = 5
                    count = 1
                    x = re_i + 1
                    y = re_j - 1
                    flag_down_end = False
                    flag_up_end = False
                    while y >= 0 and x < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        y = y - 1
                        x = x + 1
                    if y >= 0 and x < self.chessboard_size and chessboard[x][y] == 0:
                        flag_down_end = True
                    x = re_i - 1
                    y = re_j + 1
                    while x >= 0 and y < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
                        count = count + 1
                        y = y + 1
                        x = x - 1
                    if x >= 0 and y < self.chessboard_size and chessboard[x][y] == 0:
                        flag_up_end = True
                    if flag_up_end and flag_down_end:  # 说明是活棋
                        '''if count == 1:
                            score = score + chessboard[i][j] * base
                        elif count == 2:
                            score = score + chessboard[i][j] * initial * base
                        else:
                            for k in range(0, count - 2):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                        score = score + chessboard[i][j] * count * count
                    elif flag_down_end or flag_up_end:
                        '''if count == 2:
                            score = score + chessboard[i][j] * base
                        elif count == 3:
                            score = score + chessboard[i][j] * initial * base
                        elif count > 3:
                            for k in range(0, count - 3):
                                initial = initial * (initial + 1)
                                initial = initial + 1
                            score = score + chessboard[i][j] * initial * base'''
                        score = score + chessboard[i][j] * count * count/4
        return score

    def end(self, chessboard, i, j):
        count = 1
        x = i
        y = j-1
        while y >= 0 and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y - 1
        x = i
        y = j+1
        while y < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y + 1
        if count >= 5:
            return True

        count = 1
        x = i-1
        y = j
        while x >= 0 and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            x = x - 1
        x = i + 1
        y = j
        while x < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            x = x + 1
        if count >= 5:
            return True

        count = 1
        x = i - 1
        y = j - 1
        while y >= 0 and x >= 0 and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y - 1
            x = x - 1
        x = i + 1
        y = j + 1
        while x < self.chessboard_size and y < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y + 1
            x = x + 1
        if count >= 5:
            return True

        count = 1
        x = i + 1
        y = j - 1
        while y >= 0 and x < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y - 1
            x = x + 1
        x = i - 1
        y = j + 1
        while x >= 0 and y < self.chessboard_size and chessboard[x][y] == chessboard[i][j]:
            count = count + 1
            y = y + 1
            x = x - 1
        if count >= 5:
            return True
        return False


class Node:
    def __init__(self):
        self.child = []
        self.best = []
        self.point = (0, 0)
        self.mark = 0

    def set_point(self, p):
        self.point = copy.deepcopy(p)

    def add_child(self, node):
        self.child.append(node)

    def get_last_child(self):
        return self.child[-1]

    def set_best(self, node):
        self.best.clear()
        self.best.append(node)

    def get_best(self):
        return self.best


def main():
    play = AI(15, -1, 5)
    chessboard = [[0 for x in range(15)] for y in range(15)]
    play.go(chessboard)
    chessboard[7][7] = -1
    chessboard[7][8] = 1
    play.go(chessboard)



if __name__ == '__main__':
    main()