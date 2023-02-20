# Value Iteration Algorithm
import numpy as np

# define the reward matrix
R = np.array([[0,0,0], [0,10,0], [0,0,0]])

# define the discount factor
gamma = 0.9

# define the initial value function
V = np.zeros((3,3))

# define the probability
p = 0.8

# define the epsilon
epsilon = 0.0001

def value_iteration():
    while True:
        # store the old value function
        V_old = np.copy(V)
        # go through each state
        for i in range(3):
            for j in range(3):
                # calculate the expected value of each action
                action_val = []
                if i > 0:
                    action_val.append(p * (R[i-1, j] + gamma * V_old[i-1, j]))
                if i < 2:
                    action_val.append(p * (R[i+1, j] + gamma * V_old[i+1, j]))
                if j > 0:
                    action_val.append(p * (R[i, j-1] + gamma * V_old[i, j-1]))
                if j < 2:
                    action_val.append(p * (R[i, j+1] + gamma * V_old[i, j+1]))
                action_val.append(p * (R[i, j] + gamma * V_old[i, j]))
                # find the best action
                V[i, j] = max(action_val)
        # check if the values are converging
        if np.sum(np.abs(V - V_old)) <= epsilon:
            break
    # find the optimal policy
    pi = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            action_val = []
            if i > 0:
                action_val.append(p * (R[i-1, j] + gamma * V[i-1, j]))
            if i < 2:
                action_val.append(p * (R[i+1, j] + gamma * V[i+1, j]))
            if j > 0:
                action_val.append(p * (R[i, j-1] + gamma * V[i, j-1]))
            if j < 2:
                action_val.append(p * (R[i, j+1] + gamma * V[i, j+1]))
            action_val.append(p * (R[i, j] + gamma * V[i, j]))
            # find the best action
            pi[i, j] = np.argmax(action_val)
    # print the optimal value function
    print("Optimal value function:")
    print(V)
    # print the optimal policy
    print("Optimal policy:")
    print(pi)

if __name__ == '__main__':
    value_iteration()