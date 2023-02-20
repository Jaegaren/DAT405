# Value Iteration Algorithm
import numpy as np


# define the discount factor
gamma = 0.8

# define the probability
prob_moving = 0.8

prob_staying = 1 - prob_moving
# define the epsilon

epsilon = 0.001


def value_iteration():
    R = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)
    V_old = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)  # store the old value function
    diff = float('inf')
    while diff > epsilon:
        V_new = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
        for i in range(3):  # go through each state
            for j in range(3):
                action_val = []  # temporary for creating the transition values
                if i < 2:
                    action_val.append(prob_moving  * (R[i + 1, j] + prob_staying  * V_old[i, j]))  # south
                if i > 0:
                    action_val.append(prob_moving * (R[i - 1, j] + prob_staying  * V_old[i, j]))  # north
                if j < 2:
                    action_val.append(prob_moving * (R[i, j + 1] + prob_staying  * V_old[i, j]))  # east
                if j > 0:
                    action_val.append(prob_moving  * (R[i, j - 1] + prob_staying  * V_old[i, j]))  # west
                V_new[i][j] = max(action_val)
        diff = np.sum(np.abs(V_new - R))
        print(R)
        R = V_new
    return R


if __name__ == '__main__':
    print(value_iteration())
