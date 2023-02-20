# Value Iteration Algorithm
import numpy as np

# define the discount factor
gamma = 0.9

# length of the matrix
length = 3

# width of the matrix
width = 3

# define the reward matrix
R = np.zeros((length, width))
R[1, 1] = 10

# define the initial value function
#V = np.zeros((length, width))

# define the initial value function
#V = np.ones((length, width))

# define the initial value function
V = np.ones((length, width))
V[0, 1] = 500
V[2, 0] = 9
V[1, 0] = np.pi*542

# define the probability
prob_moving = 0.8
prob_staying = 1 - prob_moving


# define the epsilon
epsilon = 0.0001

def value_iteration():
    while True:
        V_old = np.copy(V)
        print(V_old)
        for i in range(length):
            for j in range(width):
                action_val = []
                if i > 0:
                    # checks if north is possible
                    action_val.append(prob_moving * (R[i - 1, j] + gamma * V_old[i - 1, j]))
                if i < length - 1:
                    # checks if south is possible
                    action_val.append(prob_moving * (R[i + 1, j] + gamma * V_old[i + 1, j]))
                if j > 0:
                    # checks if west is possible
                    action_val.append(prob_moving * (R[i, j - 1] + gamma * V_old[i, j - 1]))
                if j < width - 1:
                    # checks if east is possible
                    action_val.append(prob_moving * (R[i, j + 1] + gamma * V[i, j + 1]))
                action_val.append(prob_moving * (R[i, j] + gamma * V[i, j]))
                V[i, j] = max(action_val)
        if np.sum(np.abs(V - V_old)) <= epsilon:
            break
    pi = np.zeros((length, width))

    for i in range(length):
        for j in range(width):
            action_val = []
            if i > 0:
                action_val.append(prob_moving * (R[i - 1, j] + gamma * V[i - 1, j]))
            if i < length - 1:
                action_val.append(prob_moving * (R[i + 1, j] + gamma * V[i + 1, j]))
            if j > 0:
                action_val.append(prob_moving * (R[i, j - 1] + gamma * V[i, j - 1]))
            if j < width - 1:
                action_val.append(prob_moving * (R[i, j + 1] + gamma * V[i, j + 1]))
            action_val.append(prob_moving * (R[i, j] + gamma * V[i, j]))

            pi[i, j] = action_val.index(max(action_val))

            print("Optimal value function:")
            print(V)
            print("Optimal policy:")
            print(pi)

if __name__ == "__main__":
    value_iteration()