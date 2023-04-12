# Code below copied from Dr.Zarabi's Algorithm Design course


import numpy as np
from math import inf

class LPSolver(object):
    EPS = 1e-9
    NEG_INF = -inf

    def __init__(self, A, b, c):
        self.m = len(b)
        self.n = len(c)
        self.N = [0] * (self.n + 1)
        self.B = [0] * self.m
        self.D = [[0 for i in range(self.n + 2)] for j in range(self.m + 2)]
        self.D = np.array(self.D, dtype=np.float64)
        for i in range(self.m):
            for j in range(self.n):
                self.D[i][j] = A[i][j]
        for i in range(self.m):
            self.B[i] = self.n + i
            self.D[i][self.n] = -1
            self.D[i][self.n + 1] = b[i]
        for j in range(self.n):
            self.N[j] = j
            self.D[self.m][j] = -c[j]
        self.N[self.n] = -1
        self.D[self.m + 1][self.n] = 1

    def Pivot(self, r, s):
        D = self.D
        B = self.B
        N = self.N
        inv = 1.0 / D[r][s]
        dec_mat = np.matmul(D[:, s:s+1], D[r:r+1, :]) * inv
        dec_mat[r, :] = 0
        dec_mat[:, s] = 0
        self.D -= dec_mat
        self.D[r, :s] *= inv
        self.D[r, s+1:] *= inv
        self.D[:r, s] *= -inv
        self.D[r+1:, s] *= -inv
        self.D[r][s] = inv
        B[r], N[s] = N[s], B[r]

    def Simplex(self, phase):
        m = self.m
        n = self.n
        D = self.D
        B = self.B
        N = self.N
        x = m + 1 if phase == 1 else m
        while True:
            s = -1
            for j in range(n + 1):
                if phase == 2 and N[j] == -1:
                    continue
                if s == -1 or D[x][j] < D[x][s] or D[x][j] == D[x][s] and N[j] < N[s]:
                    s = j
            if D[x][s] > -self.EPS:
                return True
            r = -1
            for i in range(m):
                if D[i][s] < self.EPS:
                    continue
                if r == -1 or D[i][n + 1] / D[i][s] < D[r][n + 1] / D[r][s] or (D[i][n + 1] / D[i][s]) == (D[r][n + 1] / D[r][s]) and B[i] < B[r]:
                    r = i
            if r == -1:
                return False
            self.Pivot(r, s)

    def Solve(self):
        m = self.m
        n = self.n
        D = self.D
        B = self.B
        N = self.N
        r = 0
        for i in range(1, m):
            if D[i][n + 1] < D[r][n + 1]:
                r = i
        if D[r][n + 1] < -self.EPS:
            self.Pivot(r, n)
            if not self.Simplex(1) or D[m + 1][n + 1] < -self.EPS:
                return self.NEG_INF, None
            for i in range(m):
                if B[i] == -1:
                    s = -1
                    for j in range(n + 1):
                        if s == -1 or D[i][j] < D[i][s] or D[i][j] == D[i][s] and N[j] < N[s]:
                            s = j
                    self.Pivot(i, s)
        if not self.Simplex(2):
            return self.NEG_INF, None
        x = [0] * self.n
        for i in range(m):
            if B[i] < n:
                x[B[i]] = D[i][n + 1]
        return D[m][n + 1], x


def find_pure_nash(matrix):
    min_list = []
    max_list = []
    transpose = [list(x) for x in zip(*matrix)]
    for indx, i in enumerate(transpose):
        max_element = max(i)
        for index, j in enumerate(i):
            if j == max_element:
                max_list.append((index, indx))
    for indx, i in enumerate(matrix):
        min_element = min(i)
        for index, j in enumerate(i):
            if j == min_element:
                min_list.append((indx, index))
    same = [element for element in max_list if element in min_list]
    return len(same), same


def find_sum(x):
    return np.sum(x)


def find_mixed_nash(matrix):
    # Define the payoff matrix
    A = matrix.copy()
    n, m = A.shape
    row_dup = np.tile(np.array(A[0]), (n, 1))
    final_matrix = A - row_dup
    final_matrix[0] = [1 for i in range(n)]
    # print("final matrix : \n", final_matrix)
    bb = [1]
    for i in range(n - 1):
        bb.append(0)
    bounds = Bounds([0 for i in range(n)], [1 for i in range(n)])
    initial_guess = np.array([1 / n for i in range(n)])
    linear_constraint = LinearConstraint(final_matrix, bb, bb)
    res = minimize(find_sum, initial_guess, constraints=[linear_constraint], bounds=bounds)
    return res.x

def choop(x):
    return x[-1]

def mixed_nash(matrix):
    matrix = np.array(matrix)
    A = []
    b = []
    matrix = np.transpose(matrix)
    diag = np.diag([-1] * matrix.shape[0])
    for i in range(matrix.shape[0]) :
        t = np.append(np.copy(matrix[i]),[-1],axis=0)
        t = -t
        A.append(t)
        b.append(0)
        k = np.append(diag[i],[0],axis=0)
        A.append(k)
        b.append(0)
    
    k = list([1 for i in range(matrix.shape[0])])
    k.append(0)
    prob = np.array(k)
    A.append(prob)
    b.append(1)
    A.append(-prob)
    b.append(-1)
    c = list([0 for i in range(matrix.shape[0])])
    c.append(1)
    s = LPSolver(A, b, c)
    return s.Solve()[1][0:-1]


def remove_dominated(matrix):
    matrix = np.array(matrix)
    num_strategies = matrix.shape[0]
    is_dominated = np.zeros(num_strategies, dtype=bool)
    changed = True
    while changed:
        changed = False
        for i in range(num_strategies):
            if is_dominated[i]:
                continue
            for j in range(num_strategies):
                if i == j or is_dominated[j]:
                    continue
                if all(matrix[i] < matrix[j]) and any(matrix[i] < matrix[j]):
                    is_dominated[i] = True
                    changed = True
                    break
                elif all(matrix[j] < matrix[i]) and any(matrix[j] < matrix[i]):
                    is_dominated[j] = True
                    changed = True
        matrix = matrix[~is_dominated][:, ~is_dominated]
        is_dominated = is_dominated[~is_dominated]
        num_strategies = matrix.shape[0]
    return matrix


def find_nash(matrix):
    matrix = np.array(matrix)
    # while True:
    #     mm = matrix.copy()
    #     matrix = remove_dominated(matrix)
    #     n, m = matrix.shape
    #     player_2 = np.array(np.tile(np.array(list([100 for i in range(n)])), (n, 1)) - matrix).transpose()
    #     matrix = remove_dominated(player_2)
    #     n, m = matrix.shape
    #     matrix = np.array(np.tile(np.array(list([100 for i in range(n)])), (n, 1)) - matrix).transpose()
    #     if matrix.all() == mm.all():
    #         break
    # print(matrix)
    # pure = find_pure_nash(matrix)
    # print(pure[0])
    # if pure[0] > 0:
    #     print(*pure[1]) 
    pp = 100 - matrix
    pp = np.transpose(pp)
    p1 = mixed_nash(matrix)
    p2 = mixed_nash(pp)
    print(*p1)
    print(*p2)


def get_input():
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list(map(int, input().split())))
    return matrix


def test():
    matrixes = [
        # np.array([[1, 5],
        #           [2, 4]
        #           ]),
        # np.array([[90, 20],
        #           [30, 60]
        #           ]),
        # np.array([
            
        # ])
    ]
    for matrix in matrixes:
        find_nash(matrix)


if __name__ == '__main__':
    # test()
    matrix = get_input()
    find_nash(matrix)