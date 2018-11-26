import random
import time
import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


direction = [[0, 1], [1, 0], [-1, -1], [1, -1]]


# 棋形
def chess_type(c):  # c = color
    five = [[c, c, c, c, c]]
    live_four = [[0, c, c, c, c, 0]]
    chong_four = [[0, c, c, c, c, -c], [-c, c, c, c, c, 0], [c, 0, c, c, c],
                  [c, c, c, 0, c], [c, c, 0, c, c]]
    live_three = [[0, c, c, c, 0], [0, c, 0, c, c, 0], [0, c, c, 0, c, 0]]
    sleep_three = [[0, 0, c, c, c, -c], [-c, c, c, c, 0, 0], [-c, c, c, 0, c, 0],
                   [0, c, 0, c, c, -c], [0, c, c, 0, c, -c], [-c, c, 0, c, c, 0],
                   [c, c, 0, 0, c], [c, 0, 0, c, c], [c, 0, c, 0, c], [-c, 0, c, c, c, 0, -c]]
    live_two = [[0, c, c, 0, 0], [0, 0, c, c, 0], [0, c, 0, c, 0], [0, 1, 0, 0, 1, 0]]
    sleep_two = [[0, 0, 0, c, c, -c], [-c, c, c, 0, 0, 0], [-c, c, 0, c, 0, 0],
                 [0, 0, c, 0, c, -c], [-c, c, 0, 0, c, 0], [0, c, 0, 0, c, -c],
                 [c, 0, 0, 0, c]]
    all_type = [five, live_four, chong_four, live_three, sleep_three, live_two, sleep_two]
    return all_type


def count(all_type, num, chessboard):
    # 行 选定一个类型，进行查找
    for a in range(0, 7):
        type1 = all_type[a]
        for exact_type in type1:
            for i in range(15):
                num[a] = num[a] + matching(chessboard[i], exact_type)

    # 列
    chessboard_new = np.transpose(chessboard)
    for a in range(0, 7):
        type1 = all_type[a]
        for exact_type in type1:
            for i in range(15):
                num[a] = num[a] + matching(chessboard_new[i], exact_type)

    # 左对角线
    for i in range(15):
        x = i
        y = 0
        line = []
        while x < 15 and y < 15:
            line.append(chessboard[x][y])
            x = x + 1
            y = y + 1
        for a in range(0, 7):
            type1 = all_type[a]
            for exact_type in type1:
                num[a] = num[a] + matching(line, exact_type)
    for i in range(1, 15):
        x = i
        y = 0
        line = []
        while x < 15 and y < 15:
            line.append(chessboard_new[x][y])
            x = x + 1
            y = y + 1
        for a in range(0, 7):
            type1 = all_type[a]
            for exact_type in type1:
                num[a] = num[a] + matching(line, exact_type)

    # 右对角线
    for i in range(15):
        x = 0
        y = i
        line = []
        while x < 15 and y >= 0:
            line.append(chessboard[x][y])
            x = x + 1
            y = y - 1
        for a in range(0, 7):
            type1 = all_type[a]
            for exact_type in type1:
                num[a] = num[a] + matching(line, exact_type)
    for i in range(1, 15):
        x = i
        y = 14
        line = []
        while x < 15 and y >= 0:
            line.append(chessboard[x][y])
            x = x + 1
            y = y - 1
        for a in range(0, 7):
            type1 = all_type[a]
            for exact_type in type1:
                num[a] = num[a] + matching(line, exact_type)
    return num


def matching(line, type1):
    count_line = 0
    if len(line)-len(type1)+1 <= 0:
        return 0
    else:
        for i in range(0, len(line) - len(type1) + 1):
            flag = True
            for j in range(0, len(type1)):
                if line[i + j] == type1[j]:
                    continue
                else:
                    flag = False
                    break
            if flag:
                count_line = count_line + 1
        return count_line


def evaluate(chessboard, color, mycolor):
    score = 0
    all_type = chess_type(color)
    num = [0, 0, 0, 0, 0, 0, 0]
    num_after = count(all_type, num, chessboard)
    # 0 成五；1 活四；2 冲四；3 活三；4 眠三；5 活二；6 眠二
    # 单个棋形的评分
    if num_after[0] > 0:
        score = score + 8000000
        return score
    if num_after[1] > 0:
        score = score + 300000
    if num_after[2] > 0:
        score = score + 3000
    if num_after[3] > 0:
        score = score + 3000
    if num_after[4] > 0:
        score = score + 600
    if num_after[5] > 0:
        score = score + 600
    if num_after[6] > 0:
        score = score + 200
    # 当出现两个活棋时，分数要大于单个死棋
    if num_after[1] > 1:
        score = score + 300000
    if num_after[3] > 1:
        score = score + 3000
    if num_after[5] > 1:
        score = score + 600
    # 当出现活棋和死棋组合时，仅考虑相邻相差不大的情况

    if color == mycolor:
        score = score
    else:
        score = score*0.9
    return score


def not_only_one(x, y, chessboard):
    if get_point(x-1,y-1,chessboard)==0 and get_point(x-1,y,chessboard)==0 and get_point(x-1,y+1,chessboard)==0 and \
        get_point(x,y-1,chessboard)==0 and get_point(x,y+1,chessboard)==0 and get_point(x+1,y-1,chessboard)==0 and \
        get_point(x+1,y,chessboard)==0 and get_point(x+1,y+1,chessboard)==0:
        return False
    return True


def get_point(x, y, chessboard):
    if x < 0 or x > 14 or y < 0 or y > 14:
        return 0
    else:
        return chessboard[x][y]


def search(chessboard, color, candidate_list):
    max1 = -1 << 28
    max2 = -1 << 28
    next_x = 7
    next_y = 7
    candidate_list.append((next_x, next_y))
    for i in range(15):
        for j in range(15):
            if chessboard[i][j] == 0 and not_only_one(i, j, chessboard):
                chessboard[i][j] = color
                score1 = evaluate(chessboard, color, color)
                chessboard[i][j] = -color
                score2 = evaluate(chessboard, -color, color)
                chessboard[i][j] = 0
                score_max = max(score1, score2)
                if score_max > max1:
                    max1 = score_max
                    next_x = i
                    next_y = j
                    candidate_list.append((next_x, next_y))


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []

    def go(self, chessboard):
        self.candidate_list.clear()
        my_color = self.color
        search(chessboard, my_color, self.candidate_list)
