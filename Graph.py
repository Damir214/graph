import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd
import csv


class Graph:

    def __init__(self, another='not'):
        if another == 'not':
            a = nx.read_gexf(another)
        else:
            a = nx.read_gexf('ego-graph.gexf')  # Считывание файла

        self.a = a
        self.adj = nx.to_numpy_matrix(a)  # Получение матрицы смежности
        self.nodes = np.arange(len(nx.nodes(a)))  # Список всех вершин
        self.n = len(self.nodes)  # Кол-во вершин
        self.edges = nx.to_edgelist(a)  # Список смежности

        self.reversed_adj = self.adj.transpose()
        self.not_oriented = self.adj + self.reversed_adj
        self.not_oriented[np.nonzero(self.not_oriented)] = 1  # Получаем неориентированный граф

        self.tranz = self.floyd_warshall()
        self.r = self.radius()
        self.d = self.diametr()

        # self.nodes = np.arange(4)
        # self.adj = np.matrix([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0]])
        # self.not_oriented = self.adj + self.reversed_adj
        # self.not_oriented[np.nonzero(self.not_oriented)] = 1

        self.edges = []
        for i, row in enumerate(self.not_oriented):
            for j, node in enumerate(row.flat):
                if node != 0:
                    self.edges.append((i, j))
        # pos = nx.spring_layout(a)
        # per = nx.algorithms.distance_measures.periphery(a)
        # cen = nx.algorithms.distance_measures.center(a)
        # nx.draw_networkx_nodes(a, pos,
        #                        nodelist=per,
        #                        node_color='r',
        #                        node_size=50,
        #                        alpha=0.8)
        # nx.draw_networkx_nodes(a, pos,
        #                        nodelist=cen,
        #                        node_color='b',
        #                        node_size=50,
        #                        alpha=0.8)
        # nx.draw(a, pos, nodelist=list(set(pos) - (set(cen).union(set(per)))), node_color='g', node_size=50,
        #         with_labels=False)
        # plt.show()

    def neighbors(self, i, oriented=True, rev=False):
        """
        :param oriented:
        :param i: the node to get her neighbors
        :param rev: if we need neighbors of reversed graph make it True
        :return: 2-d array with indices of neighbors
        """
        if oriented:
            if rev:
                return np.nonzero(self.adj.transpose()[i])[1]
            else:
                return np.nonzero(self.adj[i])[1]
        else:
            return np.nonzero(self.not_oriented[i])[1]

    def is_weakly_connected(self):
        """
        :return: is graph weakly connected
        """
        return len(self.DFS(oriented=False)[0]) == 1

    def is_strongly_connected(self):
        """
        :return: is graph weakly connected
        """
        return len(self.kosaraju_algorithm()) == 1

    def DFS(self, start=0, order=[], oriented=True, rev=False):
        """
        Depth-first search
        :param start: start node
        :param order: order of visiting the nodes
        :param oriented: is it matter oriented or not oriented graph we check
        :param rev: if we need neighbors of reversed graph make it True
        :return: stack of visited nodes and times of entering and leaving each node
        """
        if len(order) == 0:
            not_visited = np.copy(self.nodes)
        else:
            not_visited = np.copy(order)
        stack = []
        time = [-1] * (self.n * 2)
        while len(not_visited) != 0:
            visited = []
            node = not_visited[start]
            visited, time, not_visited = self.go_down(node, time, np.delete(not_visited, 0), visited, oriented, rev)
            start = 0
            stack.append(visited)
        return stack, time

    def go_down(self, node, time, not_visited, visited, oriented, rev=False):
        """
        :param node: the start node
        :param time: the list of entrance and leaving node
        :param not_visited: the stack of not visited nodes
        :param visited: the list of visited nodes
        :param oriented: is it matter oriented or not oriented graph we check
        :param rev: if we need neighbors of reversed graph make it True
        :return: stack of nodes in weak connectivity component and times of entering and leaving each node
        """
        visited.append(node)
        time[node] = max(time) + 1
        neighbors = self.neighbors(node, oriented, rev)
        neighbors = sorted(neighbors, key=lambda x: np.argwhere(not_visited == x))
        for nb in neighbors:
            if nb in not_visited:
                visited, time, not_visited = self.go_down(nb, time, np.setdiff1d(not_visited, [nb]), visited, oriented,
                                                          rev)
        time[self.n + node] = max(time) + 1
        return visited, time, not_visited

    def BFS(self, G, s):
        """
        :param G: list of nodes
        :param s: start node
        :return: S - stack of visited nodes, P - nodes that does contain in shortest path, sigma - the length of shortest path
        """
        S = []
        P = {}
        for v in G:
            P[v] = []
        sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = dict()
        sigma[s] = 1.0
        D[s] = 0
        Q = [s]
        while Q:  # use BFS to find shortest paths
            v = Q.pop(0)
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in self.neighbors(v, oriented=False):
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # this is a shortest path, count paths
                    sigma[w] += sigmav
                    P[w].append(v)  # predecessors
        return S, P, sigma

    def accumulate_endpoints(self, betweenness, S, P, sigma, s):
        betweenness[s] += len(S) - 1
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w] + 1
        return betweenness

    def accumulate_edges(self, betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                c = sigma[v] * coeff
                if (v, w) not in betweenness:
                    betweenness[(w, v)] += c
                else:
                    betweenness[(v, w)] += c
                delta[v] += c
            if w != s:
                betweenness[w] += delta[w]
        return betweenness

    def rescale(self, betweenness):
        n = self.n
        if n < 2:
            scale = None  # no normalization
        else:
            # Scale factor should include endpoint nodes
            scale = 1 / (n * (n - 1))
        if scale is not None:
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness

    def kosaraju_algorithm(self):
        stack, times = self.DFS()
        order = sorted(range(self.n), key=lambda k: times[k + self.n])
        stack, times = self.DFS(order=order)
        return stack

    def degree_histogram(self):
        a = [0] * self.n
        for node in self.nodes:
            a[node] = len(self.neighbors(node))
        hist = [0] * max(a)
        c = Counter(a)
        s = sum(c.values())
        for i in range(len(hist)):
            hist[i] = c[i] / s
        mean = np.mean(hist) * s
        return hist, mean

    def floyd_warshall(self):
        w = np.copy(self.not_oriented)
        w = np.where(w != 0, w, np.inf)
        for k in range(self.n):
            for i, el in enumerate(w):
                for j, col in enumerate(el):
                    w[i, j] = min(w[i][j], w[i, k] + w[k, j])
        return w

    def radius(self):
        return np.min([np.max(el) for el in self.tranz])

    def diametr(self):
        return np.max(self.tranz)

    def mean_path_length(self):
        return np.mean(self.tranz)

    def center(self):
        ans = []
        for i, el in enumerate(self.tranz):
            if np.max(el) == self.r:
                ans.append(i)
        return ans

    def peripheral(self):
        ans = []
        for i, el in enumerate(self.tranz):
            if np.max(el) == self.d:
                ans.append(i)
        return ans

    def similarity_measures(self, measure='common'):
        similarity = np.zeros((self.n, self.n))

        if measure == 'common' or measure == 'jaccard':
            for i, row in enumerate(similarity):
                for j, col in enumerate(row):
                    similarity[i, j] = len([el for el in self.neighbors(i, oriented=False)
                                            if el in self.neighbors(j, oriented=False)])

            if measure == 'jaccard':
                for i, row in enumerate(similarity):
                    for j, col in enumerate(row):
                        similarity[i, j] /= len(set(np.union1d(self.neighbors(i, oriented=False),
                                                               self.neighbors(j, oriented=False))))

        elif measure == 'adamic':
            for i, row in enumerate(similarity):
                for j, col in enumerate(row):
                    neig = list(set(np.union1d(self.neighbors(i, oriented=False),
                                               self.neighbors(j, oriented=False))))
                    if len(neig) != 0:
                        if np.log(len(neig)) != 0:
                            similarity[i, j] = 1 / np.log(len(neig))
                        else:
                            similarity[i, j] = np.inf
                    else:
                        similarity[i, j] = np.inf
        elif measure == 'preferential':
            for i, row in enumerate(similarity):
                for j, col in enumerate(row):
                    similarity[i, j] = len(self.neighbors(i, oriented=False)) * len(self.neighbors(j, oriented=False))
        return similarity

    def centrality_measure(self, type_centrality='degree'):
        centrality = dict.fromkeys(self.nodes, 0.0)
        if type_centrality == 'degree':
            for node in self.nodes:
                centrality[node] = 2 * len(self.neighbors(node)) / (self.n - 1)

        elif type_centrality == 'closeness':
            for node in self.nodes:
                centrality[node] = ((sum(self.tranz[node]) - self.tranz[node][node]) ** (-1)) * (self.n - 1)
        elif type_centrality == 'betweenness':

            betweenness = dict.fromkeys(self.nodes, 0.0)  # b[v]=0 for v in G
            for s in self.nodes:
                # single source shortest paths
                S, P, sigma = self.BFS(self.nodes, s)
                # accumulation
                betweenness = self.accumulate_endpoints(betweenness, S, P, sigma, s)
            # rescaling
            betweenness = self.rescale(betweenness)
            return betweenness
        elif type_centrality == 'eigenvector':
            eig = np.linalg.eig(self.not_oriented)
            cur = eig[1][:, [eig[0].argmax(axis=0)]]
            for node in self.nodes:
                centrality[node] = complex(cur[node]).real
        elif type_centrality == 'edge_betweenness':
            betweenness = dict.fromkeys(self.nodes, 0.0)  # b[v]=0 for v in G
            # b[e]=0 for e in G.edges()
            betweenness.update(dict.fromkeys(self.edges, 0.0))
            for s in self.nodes:
                # single source shortest paths
                S, P, sigma = self.BFS(self.nodes, s)
                # accumulation
                betweenness = self.accumulate_edges(betweenness, S, P, sigma, s)
            # rescaling
            for n in self.nodes:  # remove nodes to only return edges
                del betweenness[n]
            betweenness = self.rescale(betweenness)
            return betweenness
        return centrality


graph = Graph()
pos = nx.spring_layout(graph.a, seed=8)
nx.draw(graph.a, pos, node_size=85)
plt.title('Граф')
plt.show()

print('Is weakly connected?', graph.is_weakly_connected())
print('Is strongly connected?', graph.is_strongly_connected())

hist, mean = graph.degree_histogram()
center = list(range(0, len(hist)))
plt.bar(center, hist)
plt.title('Гистограмма плотности вероятности распределения степеней вершин')
plt.show()
print('Mean node degree is - ', mean)

print('Mean path length is - ', graph.mean_path_length())
print('Radius is  - ', graph.radius())
print('Diameter is - ', graph.diametr())
print('Center nodes are - ', graph.center())
print('Peripheral nodes are - ', graph.peripheral())

for el in ['common', 'jaccard', 'adamic', 'preferential']:
    print('Metric is - ', el)
    mes = graph.similarity_measures(el)
    print(mes)
    pd.DataFrame(mes).to_csv(el + '.csv')

for metric in ['degree', 'closeness', 'betweenness', 'eigenvector']:
    res = graph.centrality_measure(metric)
    if type(res) is dict:
        plt.title(metric)
        nx.draw(graph.a, pos, node_color=list(res.values()), node_size=50,
                with_labels=False, cmap=plt.cm.Reds)
    else:
        plt.title(metric)
        nx.draw(graph.a, pos, node_color=list(res)[0], node_size=50,
                with_labels=False, cmap=plt.cm.Reds)
    plt.show()

# print(sorted(graph.centrality_measure('eigenvector').values()))
# print(sorted(nx.eigenvector_centrality_numpy(graph.a).values()))

res = graph.centrality_measure('edge_betweenness')
plt.title('edge_betweenness')
nx.draw(graph.a, pos, edge_color=list(res.values())[:-1], node_size=50, node_color='Blue',
        with_labels=False, edge_cmap=plt.cm.Reds)
plt.show()

# z = []
# Запись списков смежностей
# for el in graph.nodes:
#     cur = []
#     for ng in graph.neighbors(el):
#         cur.append(ng)
#     z.append(cur)
# print(z)
# with open('list.csv', 'w') as f:
#     writer = csv.writer(f)
#     for i in z:
#         writer.writerow(i)
