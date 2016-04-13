from __future__ import division, print_function

import networkx as nx
import numpy as np

from utilities import *

__all__ = ['SafeMDP', 'link_graph_and_safe_set', 'reachable_set',
           'returnable_set']


class SafeMDP(object):
    def __init__(self, graph, gp, S_hat0, h, L, beta=2):
        super(SafeMDP, self).__init__()
        # Scalar for gp confidence intervals
        self.beta = beta

        # Threshold
        self.h = h

        # Lipschitz constant
        self.L = L

        # GP model
        self.gp = gp

        self.graph = graph
        self.graph_reverse = self.graph.reverse()

        num_nodes = self.graph.number_of_nodes()
        num_edges = max_out_degree(graph)
        safe_set_size = (num_nodes, num_edges + 1)

        self.reach = np.empty(safe_set_size, dtype=np.bool)
        self.G = np.empty(safe_set_size, dtype=np.bool)

        self.S_hat = S_hat0
        self.S_hat0 = self.S_hat.copy()
        self.initial_nodes = self.S_hat0[:, 0].nonzero()[0].tolist()

    def compute_S_hat(self):
        """Compute the safely reachable set given the current safe_set."""
        self.reach[:] = False
        reachable_set(self.graph, self.initial_nodes, self.S, out=self.reach)

        self.S_hat[:] = False
        returnable_set(self.graph, self.graph_reverse, self.initial_nodes,
                       self.reach, out=self.S_hat)

    def add_gp_observations(self, x_new, y_new):
        """Add observations to the gp mode."""
        # Update GP with observations
        self.gp.set_XY(np.vstack((self.gp.X,
                                  x_new)),
                       np.vstack((self.gp.Y,
                                  y_new)))


def link_graph_and_safe_set(graph, safe_set):
    """Link the safe set to the graph model.

    Parameters
    ----------
    graph: nx.DiGraph()
    safe_set: np.array
        Safe set. For each node the edge (i, j) under action (a) is linked to
        safe_set[i, a]
    """
    for node, next_node in graph.edges_iter():
        edge = graph[node][next_node]
        edge['safe'] = safe_set[node:node+1, edge['action']]


def reachable_set(graph, initial_nodes, safe_set=None, out=None):
    """
    Compute the safe, reachable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to.
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the reachable set.
    safe_set: np.array
        Boolean array which on element (i,j) indicates whether taking
        action j at node i is safe.
        i=0 is interpreted as the node without taking an action.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    reachable_set: np.array
        Boolean array that indicates whether a node belongs to the reachable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if safe_set is not None:
        graph = graph.copy()
        link_graph_and_safe_set(graph, safe_set)

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),
                            max_out_degree(graph) + 1),
                           dtype=np.bool)
    else:
        visited = out

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)

    # TODO: rather than checking if things are safe, specify a safe subgraph?
    while stack:
        node = stack.pop(0)
        # iterate over edges going away from node
        for _, next_node, data in graph.edges_iter(node, data=True):
            action = data['action']
            if not visited[node, action] and data['safe']:
                visited[node, action] = True
                if not visited[next_node, 0]:
                    stack.append(next_node)
                    visited[next_node, 0] = True
    if out is None:
        return visited


def returnable_set(graph, reverse_graph, initial_nodes,
                   safe_set=None, out=None):
    """
    Compute the safe, returnable set of a graph

    Parameters
    ----------
    graph: nx.DiGraph
        Directed graph. Each edge must have associated action metadata,
        which specifies the action that this edge corresponds to.
    reverse_graph: nx.DiGraph
        The reversed directed graph, `graph.reverse()`
    initial_nodes: list
        List of the initial, safe nodes that are used as a starting point to
        compute the returnable set.
    safe_set: np.array
        Boolean array which on element (i,j) indicates whether taking
        action j at node i is safe.
        i=0 is interpreted as the node without taking an action.
    out: np.array
        The array to write the results to. Is assumed to be False everywhere
        except at the initial nodes

    Returns
    -------
    returnable_set: np.array
        Boolean array that indicates whether a node belongs to the returnable
        set.
    """

    if not initial_nodes:
        raise AttributeError('Set of initial nodes needs to be non-empty.')

    if safe_set is not None:
        graph = graph.copy()
        link_graph_and_safe_set(graph, safe_set)

    if out is None:
        visited = np.zeros((graph.number_of_nodes(),
                            max_out_degree(graph) + 1),
                           dtype=np.bool)
    else:
        visited = out

    # All nodes in the initial set are visited
    visited[initial_nodes, 0] = True

    stack = list(initial_nodes)

    # TODO: rather than checking if things are safe, specify a safe subgraph?
    while stack:
        node = stack.pop(0)
        # iterate over edges going into node
        for _, prev_node in reverse_graph.edges_iter(node):
            data = graph.get_edge_data(prev_node, node)
            if not visited[prev_node, data['action']] and data['safe']:
                visited[prev_node, data['action']] = True
                if not visited[prev_node, 0]:
                    stack.append(prev_node)
                    visited[prev_node, 0] = True
    if out is None:
        return visited