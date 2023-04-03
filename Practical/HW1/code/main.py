from scipy.optimize import *
import numpy as np


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


def mixed_nash(matrix):
    player_1 = np.array(matrix.copy())
    n, m = player_1.shape
    player_2 = np.array(np.tile(np.array(list([100 for i in range(n)])), (n, 1)) - player_1).transpose()
    print(find_mixed_nash(player_2))
    print(find_mixed_nash(player_1))


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
    while True:
        mm = matrix.copy()
        matrix = remove_dominated(matrix)
        n, m = matrix.shape
        player_2 = np.array(np.tile(np.array(list([100 for i in range(n)])), (n, 1)) - matrix).transpose()
        matrix = remove_dominated(player_2)
        n, m = matrix.shape
        matrix = np.array(np.tile(np.array(list([100 for i in range(n)])), (n, 1)) - matrix).transpose()
        if matrix.all() == mm.all():
            break
    print(matrix)
    # print(find_pure_nash(matrix))
    mixed_nash(matrix)


def get_input():
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list(map(int, input().split())))
    return matrix


def test():
    matrixes = [
        np.array([[1, 5],
                  [2, 4]
                  ]),
        np.array([[90, 20],
                  [30, 60]
                  ])
    ]
    for matrix in matrixes:
        find_nash(matrix)


if __name__ == '__main__':
    # test()
    matrix = get_input()
    find_nash(matrix)
