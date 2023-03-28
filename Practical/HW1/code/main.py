import scipy.optimize


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


def find_mixed_nash(matrix):
    pass


def find_nash(matrix):
    # print(find_pure_nash(matrix))
    print(find_mixed_nash(matrix))


def get_input():
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list(map(int, input().split())))
    return matrix


if __name__ == '__main__':
    matrix = get_input()
    find_nash(matrix)
