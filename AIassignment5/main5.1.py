# Value Iteration Algorithm
import numpy as np

# define the discount factor
gamma = 0.9

# define the probability
prob_moving = 0.8

prob_staying = 1 - prob_moving
# define the epsilon

epsilon = 0.001


def value_iteration():
    curMatrix = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)  # store the old value function
    prevMatrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    count = 0
    diff = float('inf')
    V_new = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    # first iteration is special because no backtracking
    for i in range(3):  # go through each state
        for j in range(3):
            action_val = []  # temporary for creating the transition values
            if i < 2:
                action_val.append((prob_moving * (curMatrix[i + 1, j]) + prob_staying * curMatrix[i, j]))  # south
            if i > 0:
                action_val.append((prob_moving * (curMatrix[i - 1, j]) + prob_staying * curMatrix[i, j]))  # north
            if j < 2:
                action_val.append((prob_moving * (curMatrix[i, j + 1]) + prob_staying * curMatrix[i, j]))  # east
            if j > 0:
                action_val.append((prob_moving * (curMatrix[i, j - 1]) + prob_staying * curMatrix[i, j]))  # west
            V_new[i][j] = max(action_val)
    prevMatrix = curMatrix
    curMatrix = V_new.copy()
    print('prevMatrix: ')
    print(prevMatrix)
    print('curMatrix: ')
    print(curMatrix)
    while diff > epsilon:
        for i in range(3):  # go through each state
            for j in range(3):
                action_val = []  # temporary for creating the transition values
                if i < 2:
                    action_val.append(
                        (prob_moving * gamma * (curMatrix[i + 1, j]) + prob_staying * gamma * prevMatrix[i, j]))  # south
                if i > 0:
                    action_val.append(
                        (prob_moving * gamma * (curMatrix[i - 1, j]) + prob_staying * gamma * prevMatrix[i, j]))  # north
                if j < 2:
                    action_val.append(
                        (prob_moving * gamma * (curMatrix[i, j + 1]) + prob_staying * gamma * prevMatrix[i, j]))  # east
                if j > 0:
                    action_val.append(
                        (prob_moving * gamma * (curMatrix[i, j - 1]) + prob_staying * gamma * prevMatrix[i, j]))  # west
                V_new[i][j] = max(action_val)
        diff = np.sum(np.abs(V_new - curMatrix))
        prevMatrix = curMatrix
        curMatrix = V_new
        print(curMatrix)

if __name__ == '__main__':
    value_iteration()

