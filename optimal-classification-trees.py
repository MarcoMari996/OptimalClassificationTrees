from pyomo.environ import *
from pyomo.opt import TerminationCondition
import numpy as np
from sklearn.datasets import *
from sklearn import tree as CART
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.style.use('fivethirtyeight')
colori = ['firebrick', 'royalblue', 'forestgreen', 'sandybrown', 'steelblue', 'olivedrab']
colors = mpl.colors.ListedColormap(colori)


def minmaxscaling(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def onehotencoder(Y):
    n = len(Y)
    classes = np.unique(Y)
    y = np.zeros((n, len(classes)))
    for i in range(n):
        for k in classes:
            if Y[i] == k:
                y[i, k] = int(1)
            else:
                y[i, k] = int(-1)
    return y


class TreeStructure:
    """
    That' s needed for having trace of the
    tree structure and for indexing Parent nodes,
    Left Anchestors, and Right Ancehstors for
    each single node.
    """

    def __init__(self, d):
        """
        :param d: depth of the tree 
        """
        self.d = d
        T = 2 ** (d + 1) - 1
        self.T = T
        self.nodes = list(i for i in range(1, T + 1))
        self.left_children = [node for node in self.nodes if node % 2 == 0]
        self.right_children = [node for node in self.nodes[1:] if node not in self.left_children]

    @staticmethod
    def find_parent(node):
        """
        input:
          @node: node whose parent is calculated
        output:
          @parent_node: the parent node `None` if node is 1 (root node)
        """
        return int(np.floor(node / 2)) if node != 1 else None

    def find_anchestors(self, node):
        anchestors = []
        current_node = node
        while self.find_parent(current_node) is not None:
            parent = self.find_parent(current_node)
            anchestors.append(parent)
            current_node = parent

        return anchestors[::-1]

    def find_left_anchestors(self, node):
        # rule: if the node is even, then its parent belongs to its left anchestors
        left_anchestors = []
        current_node = node
        while self.find_parent(current_node) is not None:
            parent = self.find_parent(current_node)
            if current_node % 2 == 0:
                left_anchestors.append(parent)
            current_node = parent

        return left_anchestors[::-1]

    def find_right_anchestors(self, node):
        right_anchestors = []
        current_node = node
        while self.find_parent(current_node) is not None:
            parent = self.find_parent(current_node)
            if current_node % 2 != 0:
                right_anchestors.append(parent)
            current_node = parent

        return right_anchestors[::-1]

    def find_nodes_in_left_path(self, node):
        left_children = []
        next_node = node * 2
        left_children.append(next_node)
        for n in left_children:
            next_node = n * 2
            if next_node < self.T:
                left_children.append(next_node)
                left_children.append(next_node + 1)

        return left_children


def OCT(X, y, D=2, alpha=1e-7, Nmin=5, timelimit=None, warmStart=False):
    # ---- Pre-processing steps ---- #
    K = len(np.unique(y))  # number of classes
    N = len(y)  # number of observations
    p = X.shape[1]
    hat_L = max(np.bincount(y) / N)  # baseline accuracy
    if X.min() < 0 or X.max() > 1:
        X = minmaxscaling(X)
    if y.shape != (N, K):
        Y = onehotencoder(y)

    tree = TreeStructure(d=D)
    T = 2 ** (D + 1) - 1
    Tb = int(np.floor(T / 2))
    Tl = int(np.floor(T / 2)) + 1
    # ---- Warm Start ---- #
    if warmStart:
        cart = CART.DecisionTreeClassifier(max_depth=D, max_leaf_nodes=2 ** D)
        cart.fit(X, y)
        feature = cart.tree_.feature
        initial_a = []
        initial_a_tmp = []
        for i in range(len(feature)):
            if feature[i] != -2 and feature[i + 1] != -2:
                initial_a.append(feature[i] + 1)
            elif feature[i] != -2 and feature[i + 1] == -2:
                initial_a_tmp.append(feature[i] + 1)
            else:
                continue
        initial_a.extend(initial_a_tmp)

        threshold = cart.tree_.threshold
        initial_b = []
        initial_b_tmp = []
        for i in range(len(threshold)):
            if threshold[i] != -2 and threshold[i + 1] != -2:
                initial_b.append(threshold[i])
            elif threshold[i] != -2 and threshold[i + 1] == -2:
                initial_b_tmp.append(threshold[i])
            else:
                continue
        initial_b.extend(initial_b_tmp)
        def initialize_a(model, t, j, init_a=initial_a):
            l = enumerate(init_a)
            if (t, j) in l:
                return 1
            else:
                return 0

        def initialize_b(model, t, init_b=initial_b):
            return init_b[t - 1]

    # ---- Defining useful parameters ---- #
    model = ConcreteModel()
    model.J = RangeSet(p)  # remember that RangeSet starts counting from 1
    model.I = RangeSet(N)
    model.K = RangeSet(K)
    model.Tb = RangeSet(Tb)  # indexing branch nodes
    model.Tl = RangeSet(Tl, T)  # indexing leaf nodes
    model.T = RangeSet(T)

    epsilon = np.array([np.abs(X[i + 1, :] - X[i, :]) for i in range(N - 1)])
    epsilon = epsilon[np.unique(np.nonzero(epsilon)[0]), :]
    epsilon = np.min(epsilon, axis=0)

    # ---- Decision Variables ---- #
    model.a = Var(model.Tb, model.J, domain=Binary) if warmStart is False else Var(model.Tb, model.J,
                                                                                   domain=Binary,
                                                                                   initialize=initialize_a)  # single feature splits # nb: this is a'
    model.b = Var(model.Tb, domain=NonNegativeReals) if warmStart is False else Var(model.Tb, domain=NonNegativeReals,
                                                                                    initialize=initialize_b)

    model.d = Var(model.Tb, domain=Binary)

    model.z = Var(model.I, model.Tl, domain=Binary)
    model.Ntk = Var(model.Tl, model.K, domain=Integers)
    model.Nt = Var(model.Tl, domain=Integers)
    model.l = Var(model.Tl, domain=Binary)
    model.c = Var(model.Tl, model.K, domain=Binary)
    model.L = Var(model.Tl, domain=NonNegativeReals)

    # ---- Objective Function ---- #
    model.obj = Objective(
        expr=(1 / hat_L) * sum(model.L[t] for t in model.Tl) + alpha * sum(model.d[t] for t in model.Tb),
        sense=minimize)

    # ---- Constraints ---- #
    model.cnstrLeaves = ConstraintList()
    for i in model.I:
        model.cnstrLeaves.add(expr=sum(model.z[i, t] for t in model.Tl) == 1)

    for t in model.Tl:
        # constraints on c[t, k]
        model.cnstrLeaves.add(expr=sum(model.c[t, k] for k in model.K) == model.l[t])
        # constraints on obs in leaves
        model.cnstrLeaves.add(expr=sum(model.z[i, t] for i in model.I) >= Nmin * model.l[t])
        # constraints on L : missclassification error
        model.cnstrLeaves.add(expr=model.Nt[t] == sum(model.z[i, t] for i in model.I))
        for k in model.K:
            model.cnstrLeaves.add(
                expr=model.Ntk[t, k] == (1 / 2) * sum((1 + Y[i - 1, k - 1]) * model.z[i, t] for i in model.I))
            model.cnstrLeaves.add(expr=model.L[t] >= model.Nt[t] - model.Ntk[t, k] - N * (1 - model.c[t, k]))
            model.cnstrLeaves.add(expr=model.L[t] <= model.Nt[t] - model.Ntk[t, k] + N * model.c[t, k])

        for i in model.I:
            model.cnstrLeaves.add(expr=model.z[i, t] <= model.l[t])

            # branch conditions
            # right branch condition
            for m in tree.find_right_anchestors(t):
                model.cnstrLeaves.add(
                    expr=sum(model.a[m, j] * X[i - 1, j - 1] for j in model.J) >= model.b[m] - (1 - model.z[i, t]))

            # left branch condition
            for m in tree.find_left_anchestors(t):
                model.cnstrLeaves.add(
                    expr=sum(model.a[m, j] * (X[i - 1, j - 1] + epsilon[j - 1]) for j in model.J) <= model.b[m] + (
                            1 + np.max(epsilon)) * (1 - model.z[i, t]))

    # - constraints on branch nodes - #
    model.cnstrBranches = ConstraintList()

    for t in model.Tb:
        for child in tree.find_nodes_in_left_path(t):
            if child > Tb:
                model.cnstrBranches.add(expr=model.l[child] <= model.d[t])
            else:
                model.cnstrBranches.add(expr=model.d[child] <= model.d[t])
        model.cnstrBranches.add(expr=sum(model.a[t, j] for j in model.J) == model.d[t])
        model.cnstrBranches.add(expr=model.b[t] <= model.d[t])
        # if t > 1:
        #     # cannot find parent of the root
        #     model.cnstrBranches.add(expr=model.d[t] <= model.d[tree.find_parent(t)])

    # ---- Solve the problem ---- #
    # solvername = 'glpk'
    solvername = 'gurobi'
    # solverpath = "/Users/marco/Desktop/Anaconda_install/anaconda3/bin/glpsol"
    solver = SolverFactory(solvername)
    solver.options['Timelimit'] = timelimit
    sol = solver.solve(model, tee=True, load_solutions=False)
    # Get a JSON representation of the solution
    sol_json = sol.json_repn()
    # Check solution status
    if sol_json['Solver'][0]['Status'] != 'ok':
        return None, []
    model.solutions.load_from(sol)
    # ---- Return Trained Parameters ---- #
    # splitting parameters
    A = np.zeros((Tb, p))
    b = np.zeros(Tb)
    for t in model.Tb:
        print('node {}'.format(t), '\t', 'applies a split? ', model.d[t]())
        A[t - 1, :] = [int(a) for a in model.a[t, :]()]
        print('a[{}, :] = '.format(t), A[t - 1, :])
        b[t - 1] = model.b[t]()
        print('b[{}] = '.format(t), b[t - 1])
    # classification of leaves
    C = np.zeros((Tl, K))
    for t in model.Tl:
        print('leaf {}'.format(t), '\n\t', 'contains points? ', model.l[t]())
        C[t - Tl, :] = [int(c) for c in model.c[t, :]()]
        print('\tpredicted class: ', C[t - Tl, :])
        print('\tpoints included:')
        print('\t', np.argwhere(np.array(model.z[:, t]()) > 0).reshape((-1,)))
    print('obj: ', model.obj())

    return A, b, C


def OCTH(X, y, D=2, alpha=1e-7, Nmin=5, timelimit=None):
    # ---- Pre-processing steps ---- #
    K = len(np.unique(y))  # number of classes
    N = len(y)  #  number of observations
    p = X.shape[1]
    hat_L = max(np.bincount(y) / N)  # baseline accuracy
    if X.min() < 0 or X.max() > 1:
        X = minmaxscaling(X)
    if y.shape != (N, K):
        Y = onehotencoder(y)

    tree = TreeStructure(d=D)
    # ---- Defining useful parameters ---- #
    model = ConcreteModel()
    model.J = RangeSet(p)  # remember that RangeSet starts counting from 1
    model.I = RangeSet(N)
    model.K = RangeSet(K)
    T = 2 ** (D + 1) - 1
    Tb = int(np.floor(T / 2))
    Tl = int(np.floor(T / 2)) + 1
    model.Tb = RangeSet(Tb)  # indexing branch nodes
    model.Tl = RangeSet(Tl, T)  # indexing leaf nodes
    model.T = RangeSet(T)
    model.mu = 0.005
    # ---- Decision Variables ---- #
    model.a = Var(model.Tb, model.J, domain=Reals, bounds=(-1, 1))  # single feature splits # nb: this is a'
    model.a_hat = Var(model.Tb, model.J)
    model.b = Var(model.Tb, domain=Reals, bounds=(-1, 1))
    model.d = Var(model.Tb, domain=Binary)
    model.s = Var(model.Tb, model.J, domain=Binary)

    model.z = Var(model.I, model.T, domain=Binary)

    model.l = Var(model.Tl, domain=Binary)
    model.c = Var(model.Tl, model.K, domain=Binary)
    model.L = Var(model.Tl, domain=NonNegativeReals)
    model.Ntk = Var(model.Tl, model.K, domain=Integers)
    model.Nt = Var(model.Tl, domain=Integers)
    # ---- Objective Function ---- #
    model.obj = Objective(expr=(1 / hat_L) * sum(model.L[t] for t in model.Tl) + alpha * sum(
        sum(model.s[t, j] for j in model.J) for t in model.Tb), sense=minimize)
    # ---- Constraints ---- #
    model.cnstrLeaves = ConstraintList()
    for i in model.I:
        model.cnstrLeaves.add(expr=sum(model.z[i, t] for t in model.Tl) == 1)

    for t in model.Tl:
        # constraints on L : missclassification error
        model.cnstrLeaves.add(expr=model.Nt[t] == sum(model.z[i, t] for i in model.I))
        for k in model.K:
            model.cnstrLeaves.add(expr=model.Ntk[t, k] == (1 / 2) * sum((1 + Y[i - 1, k - 1]) * model.z[i, t] for i in model.I))
            model.cnstrLeaves.add(expr=model.L[t] >= model.Nt[t] - model.Ntk[t, k] - N * (1 - model.c[t, k]))
            model.cnstrLeaves.add(expr=model.L[t] <= model.Nt[t] - model.Ntk[t, k] + N * model.c[t, k])
        # constraints on c[t, k]
        model.cnstrLeaves.add(expr=sum(model.c[t, k] for k in model.K) == model.l[t])
        model.cnstrLeaves.add(expr=sum(model.z[i, t] for i in model.I) >= Nmin * model.l[t])
        # constraints on obs in leaves
        for i in model.I:
            model.cnstrLeaves.add(expr=model.z[i, t] <= model.l[t])

            # branch conditions
            # right branch condition
            for m in tree.find_right_anchestors(t):
                model.cnstrLeaves.add(
                    expr=sum(model.a[m, j] * X[i - 1, j - 1] for j in model.J) >= model.b[m] - 2 * (
                            1 - model.z[i, t]))

            # left branch condition
            for m in tree.find_left_anchestors(t):
                model.cnstrLeaves.add(
                    expr=sum(model.a[m, j] * X[i - 1, j - 1] for j in model.J) + model.mu <= model.b[m] + (
                            2 + model.mu) * (1 - model.z[i, t]))

    # constraints on branch nodes #
    model.cnstrBranches = ConstraintList()
    for t in model.Tb:
        model.cnstrBranches.add(expr=sum(model.a_hat[t, j] for j in model.J) <= model.d[t])
        model.cnstrBranches.add(expr=sum(model.s[t, j] for j in model.J) >= model.d[t])
        model.cnstrBranches.add(expr=model.b[t] >= - model.d[t])
        model.cnstrBranches.add(expr=model.b[t] <= model.d[t])

        for j in model.J:
            model.cnstrBranches.add(expr=model.a_hat[t, j] >= model.a[t, j])
            model.cnstrBranches.add(expr=model.a_hat[t, j] >= - model.a[t, j])
            model.cnstrBranches.add(expr=model.a[t, j] >= - model.s[t, j])
            model.cnstrBranches.add(expr=model.a[t, j] <= model.s[t, j])
            model.cnstrBranches.add(expr=model.s[t, j] <= model.d[t])

        for child in tree.find_nodes_in_left_path(t):
            if child > Tb:
                model.cnstrBranches.add(expr=model.l[child] <= model.d[t])
            else:
                model.cnstrBranches.add(expr=model.d[child] <= model.d[t])
        # if t > 1:
        #     # cannot find parent of the root
        #     model.cnstrBranches.add(expr=model.d[t] <= model.d[tree.find_parent(t)])

        # ---- Solve the problem ---- #
        # solvername = 'glpk'
        solvername = 'gurobi'
        # solverpath = "/Users/marco/Desktop/Anaconda_install/anaconda3/bin/glpsol"
        solver = SolverFactory(solvername)
        solver.options['Timelimit'] = timelimit
        sol = solver.solve(model, tee=True, load_solutions=False)
        # Get a JSON representation of the solution
        sol_json = sol.json_repn()
        # Check solution status
        if sol_json['Solver'][0]['Status'] != 'ok':
            return None, []
        model.solutions.load_from(sol)
        # ---- Return Trained Parameters ---- #
        # splitting parameters
        A = np.zeros((Tb, p))
        b = np.zeros(Tb)
        for t in model.Tb:
            print('node {}'.format(t), '\t', 'applies a split? ', model.d[t]())
            A[t - 1, :] = [int(a) for a in model.a[t, :]()]
            print('a[{}, :] = '.format(t), A[t - 1, :])
            b[t - 1] = model.b[t]()
            print('b[{}] = '.format(t), b[t - 1])
        # classification of leaves
        C = np.zeros((Tl, K))
        for t in model.Tl:
            print('leaf {}'.format(t), '\n\t', 'contains points? ', model.l[t]())
            C[t - Tl, :] = [int(c) for c in model.c[t, :]()]
            print('\tpredicted class: ', C[t - Tl, :])
            print('\tpoints included:')
            print('\t', np.argwhere(np.array(model.z[:, t]()) > 0).reshape((-1,)))
        print('obj: ', model.obj())

        return A, b, C


class OptimalTreeClassifier:

    def __init__(self, D=2, multivariate=False, alpha=0.01, Nmin=5):
        # tree parameters
        self.D = D
        self.T = 2 ** (self.D + 1) - 1
        self.Tb = (np.floor(self.T / 2))
        self.Tl = (np.floor(self.T / 2)) + 1
        # splitting params in branch nodes
        self.A = None
        self.b = None
        # classification of leaves
        self.C = None
        # training hyperparams
        self.multivariate = multivariate
        self.alpha = alpha
        self.Nmin = Nmin

    def train(self, X, Y, timelimit=None, warmStart=False):
        if self.multivariate:
            self.A, self.b, self.C = OCTH(X, Y, D=self.D, alpha=self.alpha, Nmin=self.Nmin, timelimit=timelimit)
        else:
            self.A, self.b, self.C = OCT(X, Y, D=self.D, alpha=self.alpha, Nmin=self.Nmin, timelimit=timelimit,
                                         warmStart=warmStart)

    def predict(self, X):
        if X.min() < 0 or X.max() > 1:
            print('... normalizing X ...')
            X = minmaxscaling(X)

        n = X.shape[0]
        p = X.shape[1]
        pred_y = np.zeros(n)
        destination_leaf = dict()
        for i in range(n):
            node = 1
            while node <= self.Tb:
                if np.dot(self.A[node - 1, :], np.transpose(X[i, :])) >= self.b[node - 1]:
                    next_node = 2 * node + 1
                else:
                    next_node = 2 * node

                if next_node > self.Tb:
                    destination_leaf[i] = next_node

                node = next_node

            leaf = int(destination_leaf[i] - self.Tl)
            pred_y[i] = np.argwhere(self.C[leaf, :] == 1)

        return pred_y

    def score(self, X, y):
        n = X.shape[0]
        predicted = self.predict(X)
        accuracy = 0
        for i in range(n):
            if predicted[i] == y[i]:
                accuracy = accuracy + 1

        return accuracy / n

    def set_params(self, D=None, multivariate=None, alpha=None, Nmin=None):
        changed = False
        if D is not None:
            self.D = D
            self.T = 2 ** (self.D + 1) - 1
            self.Tb = (np.floor(self.T / 2))
            self.Tl = (np.floor(self.T / 2)) + 1
            changed = True
        if alpha is not None:
            self.alpha = alpha
            changed = True
        if multivariate is not None:
            self.multivariate = multivariate
            changed = True
        if Nmin is not None:
            self.Nmin = Nmin
            changed = True
        # invalidate trained parameters
        if changed:
            self.A = None
            self.b = None
            self.C = None

        return None


if __name__ == '__main__':
    data = load_wine()
    Y = data.target
    X = data.data

    OCTclassifier = OptimalTreeClassifier(D=2, alpha=0.05, multivariate=True)
    OCTclassifier.train(X, Y, warmStart=True, timelimit=None)

    print('\n-- Average Accuracy --\n\t{:.2f}%'.format(OCTclassifier.score(X, Y) * 100))
