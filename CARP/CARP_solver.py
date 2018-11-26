#!/usr/bin/python
import copy
import sys
import time
import random
import pprint

given = {"VERTICES": 0, "DEPOT": 0, "REQUIRED EDGES": 0, "NON-REQUIRED EDGES": 0, "VEHICLES": 0, "CAPACITY": 0,
         "TOTAL COST OF REQUIRED EDGES": 0}
cost_each = {}
demands_each = {}
distances = {}
paths = {}


class Graph:
    nodes = set()
    edges = []
    costs = {}
    demands = {}
    task = 0

    def __init__(self):
        for i in range(1, given["VERTICES"]+1):
            self.nodes.add(i)
        for i in range(0, given["VERTICES"]+1):
            self.edges.append([])

    def add_edge(self, start, end, cost, demand):
        self.edges[start].append(end)
        self.edges[end].append(start)
        self.costs[(min(start, end), max(start, end))] = cost
        self.demands[(min(start, end), max(start, end))] = demand


# 读取文件，得到设定基本变量，初始化图
def readfile(filename):
    lines = open(filename).readlines()
    for i in range(1, 8):
        line = lines[i].split(" : ")
        a = int(line[1])
        given[line[0]] = a
    my_graph = Graph()
    for i in range(9, (len(lines)-1)):
        line = lines[i].strip().split()
        start = int(line[0])
        end = int(line[1])
        cost = int(line[2])
        demand = int(line[3])
        my_graph.add_edge(start, end, cost, demand)
    return my_graph


# 对每一个点，求出每个点到其他各点的路径与长度, 准备path scanning
def dijkstra(graph, start):
    visited = {start: 0}
    path = {}
    nodes = copy.deepcopy(graph.nodes)
    while nodes:
        pre = None
        for node_temp in nodes:
            if node_temp in visited:
                if pre is None:
                    pre = node_temp
                elif visited[node_temp] < visited[pre]:
                    pre = node_temp
        if pre is None:
            break
        nodes.remove(pre)
        current = visited[pre]
        for edge in graph.edges[pre]:
            weight = current + graph.costs[(min(pre, edge), max(pre, edge))]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = pre
    return visited, path


class RouteType:
    route = []
    rest = given["CAPACITY"]
    cost = 0

    def __init__(self):
        self.cost = 0
        self.route = []
        self.rest = given["CAPACITY"]


def path_scanning_rule5(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"]+1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        while len(free) != 0:
            max_d = 0  # 当前选定边的花销
            min_d = float("inf")
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            for key in free:
                if key[0] == pos and free[key] + load <= given["CAPACITY"]:
                    if load < given["CAPACITY"]/2 and distances[given["DEPOT"]][key[1]] > max_d:
                        max_d = distances[pos][key[1]]
                        now_edge = key
                        now_load = free[key]
                        find = 1
                    if load >= given["CAPACITY"]/2 and distances[given["DEPOT"]][key[1]] < min_d:
                        min_d = distances[pos][key[1]]
                        now_edge = key
                        now_load = free[key]
                        find = 1
                if key[0] != pos and free[key] + load <= given["CAPACITY"]:
                    if load < given["CAPACITY"]/2 and distances[given["DEPOT"]][key[0]] > max_d:
                        max_d = distances[pos][key[0]]
                        now_edge = key
                        now_load = free[key]
                        find = -1
                    if load >= given["CAPACITY"]/2 and distances[given["DEPOT"]][key[0]] < min_d:
                        min_d = distances[pos][key[0]]
                        now_edge = key
                        now_load = free[key]
                        find = -1
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            if min_d == float("inf"):
                r.cost = r.cost + max_d
            else:
                r.cost = r.cost + min_d
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


# max
def path_scanning_rule1(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"]+1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        while len(free) != 0:
            max_d = 0  # 当前选定边的花销
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            dis = 0
            dis, now_edge, now_load, find = rule1(free, load, pos, dis, now_edge, now_load, find, max_d)
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            r.cost = r.cost + dis
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


# max
def path_scanning_rule3(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"]+1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        while len(free) != 0:
            max_d = 0  # 当前选定边的花销
            dis = 0
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            dis, now_edge, now_load, find = rule3(free, load, pos, dis, now_edge, now_load, find, max_d)
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            r.cost = r.cost + dis
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


#min
def path_scanning_rule4(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"]+1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        while len(free) != 0:
            min_d = float("inf")
            dis = 0
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            dis, now_edge, now_load, find = rule4(free, load, pos, dis, now_edge, now_load, find, min_d)
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            r.cost = r.cost + dis
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


#min
def path_scanning_rule2(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"]+1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        while len(free) != 0:
            min_d = float("inf")
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            dis = 0
            dis, now_edge, now_load, find = rule2(free, load, pos, dis, now_edge, now_load, find, min_d)
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            r.cost = r.cost + dis
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


def rule1(free, load, pos, dis, now_edge, now_load, find, max_d):
    for key in free:
        if key[0] == pos and free[key] + load <= given["CAPACITY"]:
            if distances[given["DEPOT"]][key[1]] > max_d:
                max_d = distances[given["DEPOT"]][key[1]]
                dis = distances[pos][key[1]]
                now_edge = key
                now_load = free[key]
                find = 1
        if key[0] != pos and free[key] + load <= given["CAPACITY"]:
            if distances[given["DEPOT"]][key[0]] > max_d:
                max_d = distances[given["DEPOT"]][key[0]]
                dis = distances[pos][key[0]]
                now_edge = key
                now_load = free[key]
                find = -1
    return dis, now_edge, now_load, find


def rule2(free, load, pos, dis, now_edge, now_load, find, min_d):
    for key in free:
        if key[0] == pos and free[key] + load <= given["CAPACITY"]:
            if distances[given["DEPOT"]][key[1]] < min_d:
                min_d = distances[given["DEPOT"]][key[1]]
                dis = distances[pos][key[1]]
                now_edge = key
                now_load = free[key]
                find = 1
        if key[0] != pos and free[key] + load <= given["CAPACITY"]:
            if distances[given["DEPOT"]][key[0]] < min_d:
                min_d = distances[given["DEPOT"]][key[0]]
                dis = distances[pos][key[0]]
                now_edge = key
                now_load = free[key]
                find = -1
    return dis, now_edge, now_load, find


def rule3(free, load, pos, dis, now_edge, now_load, find, max_d):
    for key in free:
        if key[0] == pos and free[key] + load <= given["CAPACITY"]:
            if free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))] > max_d:
                max_d = free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))]
                dis = distances[pos][key[1]]
                now_edge = key
                now_load = free[key]
                find = 1
        if key[0] != pos and free[key] + load <= given["CAPACITY"]:
            if free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))] > max_d:
                max_d = free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))]
                dis = distances[pos][key[0]]
                now_edge = key
                now_load = free[key]
                find = -1
    return dis, now_edge, now_load, find


def rule4(free, load, pos, dis, now_edge, now_load, find, min_d):
    for key in free:
        if key[0] == pos and free[key] + load <= given["CAPACITY"]:
            if free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))] < min_d:
                min_d = free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))]
                dis = distances[pos][key[1]]
                now_edge = key
                now_load = free[key]
                find = 1
        if key[0] != pos and free[key] + load <= given["CAPACITY"]:
            if free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))] < min_d:
                min_d = free[key] / cost_each[(min(key[0], key[1]), max(key[0], key[1]))]
                dis = distances[pos][key[0]]
                now_edge = key
                now_load = free[key]
                find = -1
    return dis, now_edge, now_load, find


def random_path_scanning(graph):
    k = 0
    free = {}  # 所有要求的边
    for i in range(1, given["VERTICES"] + 1):
        for j in range(0, len(graph.edges[i])):
            l = graph.edges[i][j]
            a = min(i, l)
            b = max(i, l)
            if graph.demands[(a, b)] != 0:
                free[(a, b)] = graph.demands[(a, b)]
                free[(b, a)] = graph.demands[(a, b)]  # 每条边正反都加入
    routes = []
    cost_total = 0
    while len(free) != 0:
        k = k + 1  # 记录当前次数
        pos = 1  # 当前位置
        demand = graph.demands  # 得到所有边需求集合
        r = RouteType()
        load = 0
        dis = 0
        while len(free) != 0:
            min_d = float("inf")
            max_d = 0
            now_edge = set()  # 当前选定边
            now_load = 0  # 当前选定边的需求
            find = -1
            index = random.randint(0, 3)
            if index == 0:
                dis, now_edge, now_load, find = rule1(free, load, pos, dis, now_edge, now_load, find, max_d)
            elif index == 1:
                dis, now_edge, now_load, find = rule2(free, load, pos, dis, now_edge, now_load, find, min_d)
            elif index == 2:
                dis, now_edge, now_load, find = rule3(free, load, pos, dis, now_edge, now_load, find, max_d)
            elif index == 3:
                dis, now_edge, now_load, find = rule4(free, load, pos, dis, now_edge, now_load, find, min_d)
            if now_edge == set():
                break

            load = load + now_load
            pos = now_edge[1]
            r.cost = r.cost + dis
            if find == -1:
                r.cost = r.cost + graph.costs[(min(now_edge[0], now_edge[1]), max(now_edge[0], now_edge[1]))]
            r.route.append((now_edge[0], now_edge[1]))
            del free[(now_edge[0], now_edge[1])]
            del free[(now_edge[1], now_edge[0])]
        if pos != given["DEPOT"]:
            r.cost = r.cost + distances[pos][given["DEPOT"]]
        r.rest = given["CAPACITY"] - load
        cost_total += r.cost
        routes.append(r)
    return routes, cost_total


def break_point(route):
    head = given["DEPOT"]
    break_point_list = []
    for i in range(0, len(route)):
        pos = route[i]
        if pos[0] != head:
            break_point_list.append(i)
        head = pos[1]

    return break_point_list


def double_point(route):
    double_point_list = []
    for i in range(0, (len(route) - 1)):
        pos = route[i]
        next_pos = route[i + 1]
        if pos[1] == next_pos[0]:
            double_point_list.append(i)
    return double_point_list


def score(p, routes):
    cost = 0
    over = 0
    for route in routes:
        cost += route.cost
        if route.rest < 0:
            over = over + route.rest
    score = cost + p * max(-over, 0)
    return score


def is_feasible(routes):
    for route in routes:
        if route.rest < 0:
            return False
    return True


def single_insertion(routes, score_now, p, tabu_list):
    best_score = score_now
    best_routes = []
    solutions = copy.deepcopy(routes)
    break_point_list = []
    for each_route in routes:
        break_point_list.append(break_point(each_route.route))

    for i in range(0, len(solutions)):
        solution = solutions[i]
        head = given["DEPOT"]
        for j in range(0, len(solution.route)):
            if j == len(solution.route)-1:
                rear = given["DEPOT"]
            else:
                pos = solution.route[j+1]
                rear = pos[0]

            edge_initial = solution.route[j]
            edge_remove = (min(edge_initial[0], edge_initial[1]), (max(edge_initial[0], edge_initial[1])))
            edge_remove_inverse = (edge_remove[1], edge_remove[0])
            solution_rest_before = solution.rest
            solution_cost_before = solution.cost
            solution.rest += demands_each[edge_remove]
            solution.cost = solution.cost - distances[head][edge_initial[0]] - cost_each[edge_remove] \
                            - distances[edge_initial[1]][rear] + distances[head][rear]
            del solution.route[j]

            for m in (list(range(0, i)) + list(range(i + 1, len(solutions)))):
                break_point_add = break_point_list[m]
                route_add = solutions[m]
                for n in break_point_add:
                    for edge in (edge_remove, edge_remove_inverse):
                        cost_before = route_add.cost
                        rest_before = route_add.rest
                        route_add.route.insert(n, edge)
                        if n == 0:
                            head_next = given["DEPOT"]
                            rear_next = route_add.route[1][0]
                        elif n == len(route_add.route)-1:
                            head_next = route_add.route[n-1][1]
                            rear_next = given["DEPOT"]
                        else:
                            head_next = route_add.route[n - 1][1]
                            rear_next = route_add.route[n+1][0]
                        temp_edge = (min(edge[0], edge[1]), max(edge[0], edge[1]))
                        route_add.cost = route_add.cost + distances[head_next][edge[0]] + cost_each[temp_edge] + \
                                 distances[edge[1]][rear_next] - distances[head_next][rear_next]
                        route_add.rest = route_add.rest - demands_each[temp_edge]
                        score_temp = score(p, solutions)
                        if score_temp < int(best_score) and tabu_list.__contains__(solutions) == False:
                                best_score = score_temp
                                best_routes = copy.deepcopy(solutions)
                        route_add.cost = cost_before
                        route_add.rest = rest_before
                        route_add.route.pop(n)
            solution.cost = solution_cost_before
            solution.rest = solution_rest_before
            solution.route.insert(j, edge_initial)
            head = edge_initial[1]
    return best_routes, best_score


def double_insertion(routes, score_now, p, tabu_list):
    best_score = score_now
    best_routes = []
    solutions = copy.deepcopy(routes)
    break_point_list = []
    double_point_list = []
    for each_route in routes:
        break_point_list.append(break_point(each_route.route))
        double_point_list.append(double_point(each_route.route))

    for i in range(0, len(solutions)):
        solution = solutions[i]
        double = double_point_list[i]
        for j in range(0, len(double)):
            position = double[j]
            if position == len(solution.route) - 2:
                rear = given["DEPOT"]
            else:
                pos = solution.route[position + 2]
                rear = pos[0]
            if position == 0:
                head = given["DEPOT"]
            else:
                head = solution.route[position-1][1]

            edge_initial1 = solution.route[position]
            edge_remove1 = (min(edge_initial1[0], edge_initial1[1]), (max(edge_initial1[0], edge_initial1[1])))
            edge_initial2 = solution.route[position+1]
            edge_remove2 = (min(edge_initial2[0], edge_initial2[1]), (max(edge_initial2[0], edge_initial2[1])))
            edge_remove_inverse1 = (edge_remove1[1], edge_remove1[0])
            edge_remove_inverse2 = (edge_remove2[1], edge_remove2[0])
            solution_rest_before = solution.rest
            solution_cost_before = solution.cost
            solution.rest = solution.cost + demands_each[edge_remove1] + demands_each[edge_remove2]
            solution.cost = solution.cost - distances[head][edge_initial1[0]] - cost_each[edge_remove1] - \
                            cost_each[edge_remove2] - distances[edge_initial2[1]][rear] + distances[head][rear]
            del solution.route[position]
            del solution.route[position]

            for m in (list(range(0, i)) + list(range(i+1, len(solutions)))):
                break_point_add = break_point_list[m]
                route_add = solutions[m]
                for n in break_point_add:
                    for edge in ((edge_remove1, edge_remove2), (edge_remove_inverse1, edge_remove_inverse2)):
                        cost_before = route_add.cost
                        rest_before = route_add.rest
                        route_add.route.insert(n, edge[0])
                        route_add.route.insert(n+1, edge[1])
                        if n == 0:
                            head_next = given["DEPOT"]
                            rear_next = route_add.route[2][0]
                        elif n == len(route_add.route)-2:
                            head_next = route_add.route[n-1][1]
                            rear_next = given["DEPOT"]
                        else:
                            head_next = route_add.route[n - 1][1]
                            rear_next = route_add.route[n + 2][0]
                        temp_edge1 = (min(edge[0][0], edge[0][1]), max(edge[0][0], edge[0][1]))
                        temp_edge2 = (min(edge[1][0], edge[1][1]), max(edge[1][0], edge[1][1]))
                        route_add.cost = route_add.cost + distances[head_next][edge[0][0]] + cost_each[temp_edge1] + \
                                         cost_each[temp_edge2] + distances[edge[1][1]][rear_next] \
                                         - distances[head_next][rear_next]
                        route_add.rest = route_add.rest - demands_each[temp_edge1] - demands_each[temp_edge2]
                        score_temp = score(p, solutions)
                        if score_temp < best_score and tabu_list.__contains__(solutions) == False:
                            best_score = score_temp
                            best_routes = copy.deepcopy(solutions)
                        route_add.cost = cost_before
                        route_add.rest = rest_before
                        route_add.route.pop(n)
                        route_add.route.pop(n)
            solution.cost = solution_cost_before
            solution.rest = solution_rest_before
            solution.route.insert(j, edge_initial1)
            solution.route.insert(j+1, edge_initial2)
    return best_routes, best_score


def TSA(initial_route):
    p = 1
    b_routes = initial_route
    bf_routes = initial_route
    b_score = score(p, b_routes)
    bf_score = score(p, bf_routes)
    k = 0
    k_b = 0
    k_bf = 0
    k_f = 0
    k_if = 0
    tabu_list = []
    t = given["REQUIRED EDGES"] / 2
    f_si = 1
    f_di = 1

    while True:
        flag = 0
        score_temp = 2147483647
        solution = initial_route
        remove_list = []
        for forbidden in tabu_list:
            forbidden[1] -= 1
            if forbidden[1] == 0:
                remove_list.append(forbidden)
        for remove in remove_list:
            tabu_list.remove(remove)

        while True:
            all_solution = []
            if k % f_si == 0:
                all_solution.append(single_insertion(solution, score_temp, p, tabu_list))
            if k % f_di == 0:
                all_solution.append(double_insertion(solution, score_temp, p, tabu_list))

            sorted(all_solution, key=lambda item: item[1])
            (new_solution, new_score) = all_solution[0]

            if new_solution:
                if is_feasible(new_solution):
                    new_flag = 2
                    k_if = k_if - 1
                else:
                    k_f = k_f - 1
                    new_flag = 1
                solution = new_solution
                score_temp = new_score
                if new_score < b_score:
                    b_score = new_score
                    b_routes = new_solution
                    flag = 1
                if new_flag == 2 and new_score < bf_score:
                    bf_score = new_score
                    bf_routes = new_solution
                    flag = 2
                if flag != 0:
                    break
            else:
                break
        if flag == 1:
            tabu_list.append([b_routes, t])
            k_b = 0
        elif flag == 2:
            tabu_list.append([bf_routes, t])
            k_b = 0
            k_bf = 0
        if k % 10 == 0:
            if k_f == 10:
                p = p / 2
            if k_if == 10:
                p = 2 * p
            k_f = 0
            k_if = 0
        k = k + 1
        k_b = k_b + 1
        k_bf = k_bf + 1
        k_f = k_f + 1
        k_if = k_if + 1
        if time.time() >= start + time_limit - 1:
            break
    return bf_routes, bf_score


if __name__ == '__main__':
    start = time.time()

    time_limit = 120
    seed = 1

    filename = sys.argv[1]

    time_limit = time_limit - 1
    time_limit1 = time_limit/6
    my_graph = readfile(filename)
    cost_each = copy.deepcopy(my_graph.costs)
    demands_each = copy.deepcopy(my_graph.demands)
    for i in range(1, given["VERTICES"] + 1):
        distance, path = dijkstra(my_graph, i)
        distances[i] = distance
        paths[i] = path
    initial_route = []
    initial_cost = float("inf")
    result_initial1, cost1 = path_scanning_rule1(my_graph)
    if cost1 < initial_cost:
        initial_cost = cost1
        initial_route = result_initial1
    result_initial2, cost2 = path_scanning_rule2(my_graph)
    if cost2 < initial_cost:
        initial_cost = cost2
        initial_route = result_initial2
    result_initial3, cost3 = path_scanning_rule3(my_graph)
    if cost3 < initial_cost:
        initial_cost = cost3
        initial_route = result_initial3
    result_initial4, cost4 = path_scanning_rule4(my_graph)
    if cost4 < initial_cost:
        initial_cost = cost4
        initial_route = result_initial4
    result_initial5, cost5 = path_scanning_rule5(my_graph)
    if cost5 < initial_cost:
        initial_cost = cost5
        initial_route = result_initial5
    print(initial_cost)
    for i in range(0, 3000):
        result_initial6, cost6 = random_path_scanning(my_graph)
        if cost6 < initial_cost:
            initial_cost = cost6
            initial_route = result_initial6
        if time.time()-start >= time_limit1:
            break
    print(initial_cost)
    final_routes, final_score = TSA(initial_route)
    print_result = "s "
    for i in range(len(final_routes)):
        route_now = final_routes[i].route
        if i != 0:
            print_result += ","
        print_result += "0,"
        for j in range(len(route_now)):
            print_result = print_result + "(" + str(route_now[j][0]) + "," + str(route_now[j][1]) + "),"
        print_result += "0"
    print(print_result)
    print("q", int(final_score))

    '''print_result = "s "
    for i in range(len(initial_route)):
        route_now = initial_route[i].route
        if i != 0:
            print_result += ","
        print_result += "0,"
        for j in range(len(route_now)):
            print_result = print_result + "(" + str(route_now[j][0]) + "," + str(route_now[j][1]) + "),"
        print_result += "0"
    print(print_result)
    print("q", int(initial_cost))

    if len(sys.argv) == 6:
        filename = sys.argv[1]
        time_limit = int(sys.argv[3])
        seed = int(sys.argv[5])

        time_limit = time_limit - 5
        time_limit1 = time_limit / 6
        my_graph = readfile(filename)
        cost_each = copy.deepcopy(my_graph.costs)
        demands_each = copy.deepcopy(my_graph.demands)
        for i in range(1, given["VERTICES"] + 1):
            distance, path = dijkstra(my_graph, i)
            distances[i] = distance
            paths[i] = path
        initial_route = []
        initial_cost = float("inf")
        result_initial1, cost1 = path_scanning_rule1(my_graph)
        if cost1 < initial_cost:
            initial_cost = cost1
            initial_route = result_initial1
        result_initial2, cost2 = path_scanning_rule2(my_graph)
        if cost2 < initial_cost:
            initial_cost = cost2
            initial_route = result_initial2
        result_initial3, cost3 = path_scanning_rule3(my_graph)
        if cost3 < initial_cost:
            initial_cost = cost3
            initial_route = result_initial3
        result_initial4, cost4 = path_scanning_rule4(my_graph)
        if cost4 < initial_cost:
            initial_cost = cost4
            initial_route = result_initial4
        result_initial5, cost5 = path_scanning_rule5(my_graph)
        if cost5 < initial_cost:
            initial_cost = cost5
            initial_route = result_initial5

        for i in range(0, 5000):
            result_initial6, cost6 = random_path_scanning(my_graph)
            if cost6 < initial_cost:
                initial_cost = cost6
                initial_route = result_initial6
            if time.time()-start >= time_limit1:
                break

        final_routes, final_score = TSA(initial_route)
        print_result = "s "
        for i in range(len(final_routes)):
            route_now = final_routes[i].route
            if i != 0:
                print_result += ","
            print_result += "0,"
            for j in range(len(route_now)):
                print_result = print_result + "(" + str(route_now[j][0]) + "," + str(route_now[j][1]) + "),"
            print_result += "0"
        print(print_result)
        print("q", int(final_score))'''





