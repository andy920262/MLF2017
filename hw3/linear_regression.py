import numpy as np
import random
import matplotlib.pyplot as plt

n = 1000

def sample_data():
    x = np.random.uniform(-1, 1, (1000, 2))
    y = np.sign(x[:,0]**2 + x[:,1]**2 - 0.6)
    y = np.array([i if random.random() >= 0.1 else -i for i in y])
    x = np.hstack((np.ones((1000, 1)), x, (x[:,0] * x[:,1]).reshape((1000, 1)), x**2))
    return x, y

if __name__ == '__main__':
    s = []
    for t in range(n):
        x_train, y_train = sample_data()
        x_test, y_test = sample_data()
        w = np.dot(np.linalg.pinv(x_train), y_train)
        y_pred = np.sign(np.dot(x_test, w))
        eout = (y_pred != y_test).mean()
        s.append(eout)
    s = np.array(s)
    print(np.mean(s))
    plt.hist(s, label='Avg:{:.4f}'.format(s.mean()), ec="black")
    plt.xlabel('Eout')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()
