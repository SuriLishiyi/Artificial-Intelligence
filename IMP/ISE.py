import copy
import random
import time
import sys
import numpy
import pprint


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
    for i in range(1, len(lines)):
        data = lines[i].split(" ")
        a = int(data[0])
        b = int(data[1])
        weight = float(data[2])
        my_graph.add_edge(a, b, weight)
    return my_graph


def read_seed(filename):
    lines = open(filename).readlines()
    seed = []
    for i in range(0, len(lines)):
        seed.append(int(lines[i].split()[0]))
    return seed


def main():
    start = time.time()
    '''network_file = sys.argv[2]
    seed_file = sys.argv[4]
    model = sys.argv[6]
    time_limit = int(sys.argv[8])
    seeds = read_seed(seed_file)'''
    network_file = sys.argv[1]
    model = "IC"
    time_limit = 60
    seeds = [56, 58, 53, 50, 28]
    # [52, 58, 41, 30, 39] [52, 58, 28, 48, 41]'''

    my_graph = read_network(network_file)

    '''pprint.pprint(my_graph.nodes)
    pprint.pprint(my_graph.in_edges)
    pprint.pprint(my_graph.out_edges)
    pprint.pprint(my_graph.weight)'''

    sum = 0
    iteration = 0
    if model == "LT":
        for i in range(0, 10000):
            count = LT(my_graph, seeds)
            sum = sum + count
            iteration += 1
            if time_limit - 3 < time.time() - start:
                break
    elif model == "IC":
        for i in range(0, 10000):
            count = IC(my_graph, seeds)
            sum = sum + count
            iteration += 1
            if time_limit - 3 < time.time() - start:
                break
    '''print(sum)
    print(iteration)'''
    print(sum/iteration)
    print(time.time()-start)


if __name__ == "__main__":
    main()



