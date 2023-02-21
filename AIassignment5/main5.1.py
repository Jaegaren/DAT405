# Value Iteration Algorithm
import numpy as np

# define the discount factor
gamma = 0.9

# define the probability
prob_moving = 0.8

prob_staying = 1 - prob_moving
# define the epsilon

epsilon = 0.01


def value_iteration():
    prevMatrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    curMatrix = np.array([[0, 0, 0], [0, 10, 0], [0, 0, 0]], dtype=float)
    V_new = np.zeros((3, 3))
    for k in range(10):
        for i in range(3):  # go through each state
            for j in range(3):
                action_val = []  # temporary for creating the transition values
                if i < 2:
                    action_val.append(calculatePositionValue(i + 1, j, i, j, gamma, prevMatrix, curMatrix, prob_moving))
                    # south
                if i > 0:
                    action_val.append(calculatePositionValue(i - 1, j, i, j, gamma, prevMatrix, curMatrix, prob_moving))
                    # north
                if j < 2:
                    action_val.append(calculatePositionValue(i, j + 1, i, j, gamma, prevMatrix, curMatrix, prob_moving))
                    # east
                if j > 0:
                    action_val.append(calculatePositionValue(i, j - 1, i, j, gamma, prevMatrix, curMatrix, prob_moving))
                    # west
                V_new[i][j] = max(action_val)
        prevMatrix = V_new.copy()
        print(prevMatrix)



def calculatePositionValue(directionX, directionY, x, y, gammaG, oldMatrix, currentMatrix, probMove):
    firstEquationPart = \
        probMove * ((gammaG * oldMatrix[directionX][directionY]) + currentMatrix[directionX][directionY])
    secondEquationPart = ((1 - probMove) * ((oldMatrix[x][y] * gammaG) + (currentMatrix[x][y])))
    return firstEquationPart + secondEquationPart


if __name__ == '__main__':
    value_iteration()


