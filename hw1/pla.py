import numpy as np
import matplotlib.pyplot as plt

def train(x, y, lr):
    w = np.zeros(5)
    for t in range(1, 1000):
        mistake = []
        for xi, yi in zip(x, y):
            if np.dot(w, xi) * yi <= 0:
                mistake.append((xi, yi))
        if len(mistake) == 0:
            return t
        xt, yt = mistake[np.random.randint(len(mistake))]
        w += lr * yt * xt

def read_data(path):
    x, y = [], []
    with open(path, 'r') as fd:
        for line in fd:
            line = line.strip().split('\t')
            x.append([1.0] + [float(x) for x in line[0].split(' ')])
            y.append(float(line[1]))
    return np.array(x), np.array(y)

if __name__ == '__main__':
    x, y = read_data('hw1_8_train.dat')
    ret = []
    for i in range(2000):
        ret.append(train(x, y, 1))
    ret = np.array(ret)

    plt.hist(ret, bins=range(ret.min(), ret.max() + 5, 5), label='Avg:{}'.format(ret.mean()), ec="black")
    plt.xlabel('#update')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()
