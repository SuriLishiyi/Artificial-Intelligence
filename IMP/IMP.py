import copy
import random
import time
import sys
from queue import PriorityQueue


def IC(my_graph, seed_set):
    activity_set = copy.deepcopy(seed_set)
    activity_flag = copy.deepcopy(seed_set)
    count = len(activity_set)
    while len(activity_set) != 0:
        new_activity_set = []
        for each_seed in activity_set:
            for neighbor in my_graph.out_edges[each_seed]:
                if neighbor not in activity_flag and random.random() <= my_graph.weight[(each_seed, neighbor)]:
                    new_activity_set.append(neighbor)
                    activity_flag.append(neighbor)
        count = count + len(new_activity_set)
        activity_set = new_activity_set
    return count


def LT(my_graph, seed_set):
    activity_set = copy.deepcopy(seed_set)
    all_activity_set = copy.deepcopy(seed_set)
    threshold = {}
    for node in my_graph.nodes:
        threshold[node] = random.random()
        if threshold[node] == 0:
            activity_set.append(node)
            all_activity_set.append(node)
    count = len(activity_set)
    while len(activity_set) != 0:
        new_activity_set = []
        for each_seed in activity_set:
            for neighbor in my_graph.out_edges[each_seed]:
                total_weight = 0
                for inverse_neighbor in my_graph.in_edges[neighbor]:
                    if inverse_neighbor in all_activity_set:
                        total_weight = total_weight + my_graph.weight[(inverse_neighbor, neighbor)]
                if total_weight >= threshold[neighbor]:
                    if neighbor not in all_activity_set:
                        new_activity_set.append(neighbor)
                        all_activity_set.append(neighbor)
        count = count + len(new_activity_set)
        activity_set = copy.deepcopy(new_activity_set)
    return count


class Graph:
    nodes = set()
    in_edges = []
    out_edges = []
    weight = {}

    def __init__(self, node_num):

        self.in_edges.append([])
        self.out_edges.append([])
        for i in range(1, node_num+1):
            self.nodes.add(i)
            self.in_edges.append([])
            self.out_edges.append([])

    def add_edge(self, a, b, weight):
        self.out_edges[a].append(b)
        self.in_edges[b].append(a)
        self.weight[(a, b)] = weight


def read_network(filename):
    lines = open(filename).readlines()
    num = lines[0].split(" ")
    node_num = int(num[0])
    edge_num = int(num[1])
    my_graph = Graph(node_num)
    for i in range(1, len(lines) - 1):
        data = lines[i].split(" ")
        a = int(data[0])
        b = int(data[1])
        weight = float(data[2])
        my_graph.add_edge(a, b, weight)
    return my_graph


class Job:
    def __init__(self, degree, node):
        self.degree = degree
        self.node = node
        return

    def __lt__(self, other):
        return self.degree > other.degree


'''def degree_discount(my_graph, k):
    seed_set = []
    all_nodes = copy.deepcopy(my_graph.nodes)
    degree = {}
    d_degree = PriorityQueue()
    t = {}
    for node in all_nodes:
        degree[node] = len(my_graph.edges[node]) + len(my_graph.inverse_edges[node])
        d_degree.put(Job(degree[node], node))
        t[node] = 0
    for i in range(0, k):
        temp_job = d_degree.get()
        max_node = temp_job.node
        seed_set.append(max_node)
        for neighbor in my_graph.edges[max_node]:
            if neighbor not in seed_set:
                t[neighbor] += 1
                temp = degree[neighbor] - 2*t[neighbor] - (degree[neighbor]-t[neighbor]) * t[neighbor] * 0.8
                d_degree.put(Job(temp, neighbor))
    return seed_set'''


def CELF(my_graph, k, model):
    seeds = []
    R = 100
    value = PriorityQueue()
    for node in my_graph.nodes:
        if len(my_graph.out_edges[node]) != 0:
            temp = 0
            for i in range(R):
                if model == "IC":
                    temp += IC(my_graph, seeds + [node])
                elif model == "LT":
                    temp += LT(my_graph, seeds + [node])
            temp = temp / R
            value.put(Job(temp, node))
        else:
            value.put(Job(0, node))
    max_job = value.get()
    max_node = max_job.node
    seeds.append(max_node)
    while len(seeds) < k:
        pre_job = value.get()
        pre_max_node = pre_job.node
        temp = 0
        for i in range(R):
            if model == "IC":
                temp = temp + IC(my_graph, seeds + [pre_max_node]) - IC(my_graph, seeds)
            elif model == "LT":
                temp = temp + LT(my_graph, seeds + [pre_max_node]) - LT(my_graph, seeds)
        temp = temp / R
        value.put(Job(temp, pre_max_node))
        current_job = value.get()
        if current_job.node == pre_max_node:
            seeds.append(current_job.node)
        else:
            value.put(current_job)
    return seeds


def CELF2(my_graph, k, model, pre_seeds):
    seeds = []
    R = 100
    value = PriorityQueue()
    for node in pre_seeds:
        if len(my_graph.out_edges[node]) != 0:
            temp = 0
            for i in range(R):
                if model == "IC":
                    temp += IC(my_graph, seeds + [node])
                elif model == "LT":
                    temp += LT(my_graph, seeds + [node])
            temp = temp / R
            value.put(Job(temp, node))
        else:
            value.put(Job(0, node))
    max_job = value.get()
    max_node = max_job.node
    seeds.append(max_node)
    while len(seeds) < k:
        pre_job = value.get()
        pre_max_node = pre_job.node
        temp = 0
        for i in range(R):
            if model == "IC":
                temp = temp + IC(my_graph, seeds + [pre_max_node]) - IC(my_graph, seeds)
            elif model == "LT":
                temp = temp + LT(my_graph, seeds + [pre_max_node]) - LT(my_graph, seeds)
        temp = temp / R
        value.put(Job(temp, pre_max_node))
        current_job = value.get()
        if current_job.node == pre_max_node:
            seeds.append(current_job.node)
        else:
            value.put(current_job)
    return seeds


def map_degree(my_graph, size):
    pre_seed = []
    count_degree = PriorityQueue()
    for node in my_graph.nodes:
        if len(my_graph.out_edges[node]) != 0:
            degree = len(my_graph.out_edges[node])
            count_degree.put(Job(degree, node))
    for i in range(0, size):
        job = count_degree.get()
        pre_seed.append(job.node)
    return pre_seed


def main():
    '''start = time.time()
    network_file = sys.argv[2]
    seed_size = int(sys.argv[4])
    model = sys.argv[6]
    time_limit = sys.argv[8]
    my_graph = read_network(network_file)'''

    start = time.time()
    network_file = sys.argv[1]
    model = "LT"
    time_limit = 60
    my_graph = read_network(network_file)
    seed_size = 5

    if len(my_graph.nodes) < 5000 and seed_size < 10:
        seeds = CELF(my_graph, seed_size, model)
        for seed in seeds:
            print(seed)
    elif seed_size < 10 and len(my_graph.nodes) >= 5000:
        pre = map_degree(my_graph, 100)
        seeds = CELF2(my_graph, seed_size, model, pre)
        for seed in seeds:
            print(seed)
    elif 10 <= seed_size < 30:
        pre = map_degree(my_graph, 2 * seed_size)
        seeds = CELF2(my_graph, seed_size, model, pre)
        for seed in seeds:
            print(seed)
    elif seed_size >= 30:
        seeds = map_degree(my_graph, seed_size)
        for seed in seeds:
            print(seed)
    print(time.time()-start)
    '''pre_seed = degree_discount(my_graph, seed_size * 3)
    print(pre_seed)
    seeds2 = CELF2(my_graph, seed_size, model, pre_seed)
    print(time.time() - start)
    print(seeds2)

    pre = map_degree(my_graph, seed_size)
    print(pre)
    seeds3 = CELF2(my_graph, seed_size, model, pre)
    print(time.time() - start)
    print(seeds3)'''


if __name__ == '__main__':
    main()



