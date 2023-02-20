# Value Iteration Algorithm
import numpy as np

# define the reward matrix
R = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]])

# define the discount factor
gamma = 0.9

# define the probability
prob_moving = 0.8

prob_staying = 1 - prob_moving
# define the epsilon
epsilon = 0.0001


def value_iteration():
    while True:
        V_old = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)  # store the old value function
        action_val = []
        for i in range(3):  # go through each state
            for j in range(3):
                action_val = []
                # calculate the expected value of each action
                if i < 2:
                    action_val.append(prob_moving * (R[i + 1, j] + prob_staying * V_old[i, j]))  # south
                if i > 0:
                    action_val.append(prob_moving * (R[i - 1, j] + prob_staying * V_old[i, j]))  # north
                if j < 2:
                    action_val.append(prob_moving * (R[i, j + 1] + prob_staying * V_old[i, j]))  # east
                if j > 0:
                    action_val.append(prob_moving * (R[i, j - 1] + prob_staying * V_old[i, j]))  # west
                V_old[i][j] = max(action_val)
        print(V_old)
        break
        # take highest from list, put into new matrix
        # use new matrix as old matrix

def value_iteration2(R, gamma, prob_moving, epsilon):
    V_old = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)  # store the old value function
    V_old_prev = np.copy(V_old)
    while True:
        action_val = []
        for i in range(3):  # go through each state
            for j in range(3):
                action_val = []
                # calculate the expected value of each action
                if i < 2:
                    action_val.append(prob_moving * (R[i + 1, j] + gamma * V_old[i, j]))  # south
                if i > 0:
                    action_val.append(prob_moving * (R[i - 1, j] + gamma * V_old[i, j]))  # north
                if j < 2:
                    action_val.append(prob_moving * (R[i, j + 1] + gamma * V_old[i, j]))  # east
                if j > 0:
                    action_val.append(prob_moving * (R[i, j - 1] + gamma * V_old[i, j]))  # west
                V_old[i][j] = max(action_val)
        if np.all(np.abs(V_old - V_old_prev) < epsilon):  # check convergence
            break
        else:
            V_old_prev = np.copy(V_old)
    return V_old

if __name__ == "__main__":
    #value_iteration()
    print(value_iteration2(R, 0.9, 0.8, 0.0001))