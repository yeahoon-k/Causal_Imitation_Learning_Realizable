import networkx as nx
import itertools
from src.utils.utils import pairs_to_dict_bi, pairs_to_dict, union, visited_deque
from src.utils.graphs import nx_graph
from typing import AbstractSet as ASet, Tuple, Union, FrozenSet, Collection
from src.agent.agent_utils import StructuralCausalModel as SCM
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

import pandas as pd
import numpy as np

Vars = Union[str, ASet[str]]

def _wrap(v_or_vs: Vars) -> FrozenSet[str]: # make frozenset
    return frozenset({v_or_vs}) if isinstance(v_or_vs, str) else frozenset(v_or_vs)

class Agent:
    '''
    test_g10 = nx_graph({'A', 'C', 'X', 'Y'},
                        {('C', 'X'), ('X', 'Y')},
                        {('A','X'),('A','Y'),('C','Y')})
    '''
    def __init__(self, dag: nx): #, is_expert=False):
        self.G = dag
        self.edges = set(self.G.edges)
        self.Vs = set(self.G.nodes)
        self.bidirected_edges = self.G.bidirected_edges
        self.order = self.G.order
        self._ch = pairs_to_dict(self.edges)
        self._pa = pairs_to_dict(self.edges, reverse=True)
        self._sib = pairs_to_dict_bi(dag.bidirected_edges) # ('Y':['W1','W3'], ...)
        self.ordered_topology = self.G.ordered_topology

    def bi_dir(self, V: Vars) -> FrozenSet[str]:
        """bi-directed edges of a vertex"""
        #print(self._sib[V])
        return self._sib[V]

    def pa(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ parents of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        return union(self._pa[V] for V in Vs)

    def an(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ ancestors of a vertex or vertices """
        return self.__an(_wrap(V_or_Vs))

    def __an(self, V_or_Vs: FrozenSet[str]) -> FrozenSet[str]:
        Vs = _wrap(V_or_Vs)
        if not self.pa(Vs):
            return frozenset()
        return self.__an(self.pa(Vs)) | self.pa(Vs)

    def ch(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ children of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        return union(self._ch[V] for V in Vs)

    def de(self, V_or_Vs: Vars) -> FrozenSet[str]:
        return self.__de(_wrap(V_or_Vs))

    def __de(self, V_or_Vs: Vars) -> FrozenSet[str]:
        """ descendants of a vertex or vertices """
        Vs = _wrap(V_or_Vs)
        if not self.ch(Vs):
            return frozenset()
        return self.__de(self.ch(Vs)) | self.ch(Vs)

    def before(self, X, order):
        return set(order[:order.index(X)])

    def topological_order(self, backward = False):
        gg = nx.DiGraph(self.edges)
        gg.add_nodes_from(self.Vs)
        top_to_bottom = list(nx.topological_sort(gg))
        if backward:
            return tuple(reversed(top_to_bottom))
        else:
            return tuple(top_to_bottom)

    def d_separation(self, Xs: Vars, Ys: Vars, Zs: Vars = frozenset()) -> bool:
        Xs, Ys, Zs = _wrap(Xs), _wrap(Ys), _wrap(Zs)
        Xs = Xs - Zs
        Ys = Ys - Zs
        if Xs & Ys: # not disjoint node
            return False
        if not Xs or not Ys: # no node in the sets
            return True
        return all(self.__inner_d_separation(X, Y, Zs) for X, Y in itertools.product(Xs, Ys))

    def __inner_d_separation(self, X: str, Y: str, Zs: FrozenSet[str]) -> bool:
        # For *semi-markovian graph
        # create deque instance
        q = visited_deque({(W, True) for W in self.ch(X)} |
                          {(W, False) for W in self.pa(X)} |
                          {(W, True) for W in self.bi_dir(X)})
        while q:
            W, is_child = q.pop() # W is current node
            if W == Y:  # reached to Y
                return False
            if is_child:  # if child (..->W)
                if (Zs & self.de(W)) or W in Zs:
                    q.extend({(parent, False) for parent in self.pa(W)}) #the next node is parent(False)
                    q.extend(({(sib, True) for sib in self.bi_dir(W)})) #the next node can be treated as child of W
                if W not in Zs:
                    q.extend({(child, True) for child in self.ch(W)})
            else:  # if parent(..<-W)
                if W not in Zs:
                    q.extend({(child, True) for child in self.ch(W)})
                    q.extend({(parent, False) for parent in self.pa(W)})
                    q.extend(({(sib, True) for sib in self.bi_dir(W)}))  # the next node can be treated as child of W
        return True

    def get_c_factors(self, V_or_Vs):
        """return c-components nodes of this graph(class)"""
        if isinstance(V_or_Vs, str):
            V_or_Vs = set({V_or_Vs})
        return union(self.__get_c_factors(V, set()) for V in V_or_Vs)

    def __get_c_factors(self, V, tmp):
        """inner loop function for get_c_factors which returns c-components nodes"""
        bi_di_not_visit = self.bi_dir(V) - tmp # self.bi_dir(V)
        if not bi_di_not_visit:
            return {V}
        tmp |= {V}
        for node in bi_di_not_visit:
            tmp |= self.__get_c_factors(node, tmp)
        return tmp

    def do_graph(self, Xs):
        '''
        :param Xs: intervened node
        :return: Manipulated do-graph
        '''
        Xs = _wrap(Xs)
        # Manipulated graph for do operation
        new_di_g = [(k, v) for k, v in self.edges if v not in Xs]
        new_bi_di_g = []
        for k, v in self.bidirected_edges:
            if v not in Xs and k not in Xs:
                new_bi_di_g.append((k, v))

        mod_G = Agent(nx_graph({}, new_di_g, new_bi_di_g))
        return mod_G

    def backdoor_admissible(self, Xs, Ys, Zs) -> bool:
        Xs, Ys = _wrap(Xs), _wrap(Ys)
        # Manipulated graph e.g., ( {X} -x-> {...} )
        new_g = [(k, v) for k, v in self.edges if k not in Xs]
        mod_G = Agent(nx_graph({}, new_g))
        # ID test
        return True if mod_G.d_separation(Xs, Ys, Zs) else False

    def subgraph(self, nodes):
        """Return subgraph corresponding with input nodes """
        new_edges = set()
        new_bi_di = set()
        if nodes == None:
            return self.edges, self.bidirected_edges
        for node in nodes: # [a,b,c] -> a, pa[a] => {'a' : ['b','c'], 'b':['a'], ....}
            for pa in self.pa(node):
                if pa in nodes:
                    new_edges.add((pa, node))
            for ch in self.ch(node):
                if ch in nodes:
                    new_edges.add((node, ch))
            for sib in self.bi_dir(node):
                if sib in nodes:
                    new_bi_di.add((node, sib))
                    new_bi_di.add((sib, node))
        return new_edges, new_bi_di

    def identify(self, C: set, T=None) -> bool:
        """An identify function determining if Q[C] is computable from Q[T] """
        # Summation of An(C)
        new_edges, new_bi_di = self.subgraph(T)
        mod_G = Agent(nx_graph({}, new_edges, new_bi_di))
        A = mod_G.an(C) | C
        if A == C:
            return True
        if A == T:
            return False

        # Q-decomposition
        new_edges_A, new_bi_di_A = self.subgraph(A)
        mod_G_A = Agent(nx_graph({}, new_edges_A, new_bi_di_A))
        c_components_C = mod_G_A.get_c_factors(C)
        print(c_components_C)
        return self.identify(C, c_components_C)

    # Imitation Learning with Unobserved Confounders (Junzhe Zhang, 2020)
    def imitate(self, Xs: set, Ys: set):
        """Return the policy space that can imitate expert policy"""
        # Imitate the functions of listIDspace
        list_pi = []
        for policy in self.listIDspace(Xs, Ys):
            for surrogate_S in self.listMINsep(Xs, Ys, policy):
                if self.identify(surrogate_S, policy):
                    #(surrogate_S, policy) : imitation instrument
                    list_pi.append([surrogate_S, policy])
        return list_pi

    def listMINsep(self, Xs, Ys, Zs): # !!! 논문 기반 재구현
        # find the surrogate space S (i.e., X _||_ Y | S)
        setcandiS = []  # Zs is conditioned from the "listIDspace"
        #Vs = self.Vs - Xs - Ys - Zs
        Vs = Zs - Xs - Ys
        for num_ele in range(len(Vs)+1):
            for candiS in itertools.combinations(Vs, num_ele):
                if self.d_separation(Xs, Ys, set(candiS) | Zs): #
                    setcandiS.append(set(candiS))
        return setcandiS

    def listIDspace(self, Xs: set, Ys: set):
        """Find all identifiable subspaces w.r.t. (G, PI, Y)"""
        Zs = self.Vs - Xs - Ys - self.de(Xs) # (Z = V \X \Y \de(X))
        candiZ = list()
        self.__listIDspaceHelper(Xs, Ys, set(), Zs, candiZ)
        return candiZ

    def __listIDspaceHelper(self, Xs, Ys, PIL: set, PIR: set, candiZ):
        """inner loop function for listIDspace"""
        if self.identify(Ys, PIL | Ys):
            candiZ.append(PIL)
        if PIL == PIR or len(PIR) == 0: # end of search
            return
        else:
            V = next(iter(PIR - PIL))
            self.__listIDspaceHelper(Xs, Ys, PIL | {V}, PIR | {V}, candiZ)
            self.__listIDspaceHelper(Xs, Ys, PIL - {V}, PIR - {V}, candiZ)

    # Sequential Imitation Learning with Unobserved Confounders (Daniel Kumor, 2021)
    def order_based_topology(self, ordered_by, origin_order) -> Tuple:
        """ Reorder nodes based on origin order """
        _ordered = ()
        for e in origin_order:
            if e in ordered_by:
                _ordered = _ordered + (e,)
        return _ordered

    def hasvalidAdjustment(self, Ox, Oi, Xi, Y, order=None):
        GY = self.an(Y)|Y
        G_edges, G_bi_di = self.subgraph(GY)
        G_Y_new = Agent(nx_graph({}, G_edges, G_bi_di))

        C_comps = G_Y_new.get_c_factors(Oi)

        GC_ele = G_Y_new.do_graph(G_Y_new.pa(C_comps)).an(C_comps) | C_comps
        GC_edges, GC_bi_di = G_Y_new.subgraph(GC_ele)
        GC = Agent(nx_graph({}, GC_edges, GC_bi_di))

        OC = C_comps - (Ox | {Oi})
        return GC.d_separation(Oi, OC, OC & self.before(Xi, order))


    def findOx(self, Os, X, Y, order=None):
        Ox = dict() # elements of O to X
        GY = self.an(Y)|Y
        G_edges, G_bi_di = self.subgraph(GY) # get edges from ancestor nodes of Y
        G_Y_new = Agent(nx_graph({}, G_edges, G_bi_di)) # manipulative ancestral Y graph

        while True:
            pastOx = dict(Ox)
            for Oi in G_Y_new.topological_order(True):
                # Oi의 children이 존재 & Oi의 childern이 Ox의 key에 존재
                if G_Y_new.ch(Oi) and G_Y_new.ch(Oi) <= Ox.keys():
                    Xi = self.order_based_topology({Ox[_] for _ in G_Y_new.ch(Oi)}, G_Y_new.topological_order())[0]
                    if self.hasvalidAdjustment(set(Ox.keys()), Oi, Xi, {'Y'}, order=order):
                        Ox[Oi] = Xi
                elif Oi in X and self.hasvalidAdjustment(set(Ox.keys()), Oi, Oi, {'Y'}, order=order):
                    Ox[Oi] = Oi
            if pastOx == Ox:
                break
        return set(Ox.keys())


    def get_seq_piBD(self, Xs, Y, order=None):
        Ox = self.findOx(self.Vs, Xs, Y, order)
        Xs_prime = Ox & Xs

        # markov_boundary
        n_egdes, n_bidi = self.subgraph(self.an(Y)|Y)
        dag_anY_doXs = Agent(nx_graph({}, n_egdes, n_bidi)).do_graph(Xs_prime)
        c_comp_ox = dag_anY_doXs.get_c_factors(dag_anY_doXs.ch(Ox)|Ox)
        Zs = dag_anY_doXs.pa(c_comp_ox)|c_comp_ox - Ox # markov boundary

        # boundary action
        XBs = {Xi for Xi in Xs & self.Vs if not (self.ch(Xi) <= Ox)}

        # get seq-pi-BD
        policy = {Xi: (Zs | XBs) & self.before(Xi, order) for Xi in Xs_prime}
        return policy


def get_policy(exp: int):
    """Get pi-BD policies for test"""
    # Imitator = Agent(test_msbd1)
    # policy = Imitator.get_seq_piBD({'X1', 'X2', 'X3', 'X4'}, {'Y1', 'Y2'}, Imitator.order)
    policies = {
        'control': { # from Imitator.get_seq_piBD()
            # 'X1': frozenset({}),
            # 'X2': frozenset({'Z1'}),
            # 'X3': frozenset({}),
            # 'X4': frozenset({'Z2'})
        },
        'experiment_1': { # W
            'X': frozenset({'W'}),
        },
        'experiment_2': { #
            'X1': frozenset({}),
            'X2': frozenset({'Z1'}),
            'X3': frozenset({'Z1', 'X2', 'W1'}),
            'X4': frozenset({'X3'})
        }
    }
    if exp == 1:
        return policies['control'], policies['experiment_1'] # Nothing (expert) v.s. W (graph1)
    elif exp == 2:
        return policies['control'], policies['experiment_2']
    else:
        ValueError("There is no a matched experiment")

def get_policy_l25(exp: int):
    policies = {
        'expert': {  # expert's l2.5 action
            ('X', 'Z'): 1
        },
        'experiment_1': {  # l2.5 imitator
            ('X', 'Z'): 1,
        },
    }
    if exp == 1:
        return policies['expert'], policies['experiment_1'] # expert v.s. l2.5 imitate
    else:
        ValueError("There is no a matched experiment")

def __inner_L25(graph):
    Ag = Agent(graph)
    M = SCM(Ag)

    # 1. Expert: L2.5 action을 수행하는 SCM에서의 Y 기대값
    #   - 여기서는 expert policy는 구조식 안에 이미 들어있다고 가정
    # data_expert = M.sample_binary_data(
    #     sample_size=1000,
    #     intervened=False,   # policy 개입은 없음
    #     L25_spec=get_policy_l25(1)[0],  # 대신 edge-level L2.5 개입
    # )
    # E_EY = data_expert['Y'].mean()

    # 1. Expert: L2.5 action을 수행하는 SCM에서의 Y 기대값
    alpha = 0.6  # 예: 70%는 L2.5, 30%는 L1
    n_total = 1000
    n_L25 = int(alpha * n_total)
    n_L1 = n_total - n_L25

    # 같은 M, 같은 SCM 위에서
    # # 1-(1) L2.5 regime에서 온 샘플
    # data_L25_expert = M.sample_binary_data(
    #     sample_size=n_L25,
    #     intervened=False,
    #     L25_spec=get_policy_l25(1)[0],  # L2.5 action 켠 상태
    # )

    # 1-(2) L2.5 regime에서 온 샘플
    data_L25_L1_expert = M.sample_binary_data(
        sample_size=n_L25,
        intervened=False,
        L25_spec=get_policy_l25(1)[0],  # L2.5 action 켠 상태
    )

    # 1-(1) L1 regime에서 온 샘플 (L2.5 없음) => return : L2.5 \cup L1
    data_L25_L1_expert = M.sample_binary_data(
        sample_size=n_L1,
        intervened=False,
        L25_spec=None,  # 그냥 원래 구조식만
    )

    # # 1-(2) L2.5 regime에서 온 샘플
    # data_L25_L1_expert = M.sample_binary_data(
    #     sample_size=n_L25,
    #     intervened=False,
    #     L25_spec=get_policy_l25(1)[0],  # L2.5 action 켠 상태
    # )

    E_EY = data_L25_L1_expert['Y'].mean()

    # 2. Imitator 1: pi-bd without L2.5
    data_naive = M.sample_binary_data(
        sample_size=n_total,
        intervened=True,
        policy=get_policy(1)[1],   # Pi-backdoor policy (Junzhe)
        Ys={'Y'},
        L25_spec=None  # No L2.5 strategy
    )
    I_EY_naive = data_naive['Y'].mean()

    # 3. Imitator 2: pi-bd policy + L2.5
    data_L25 = M.sample_binary_data(
        sample_size=n_total,
        intervened=True,
        policy=get_policy(1)[1], # same Pi-backdoor policy (Junzhe) as naive policy
        Ys={'Y'},
        L25_spec=get_policy_l25(1)[1]         # 여기는 L2.5 적용
    )
    I_EY_L25 = data_L25['Y'].mean()

    # 5. Expert vs Imitator's policies
    return (E_EY - I_EY_naive) ** 2, (E_EY - I_EY_L25) ** 2

