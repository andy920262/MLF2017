import numpy as np
import random
import matplotlib.pyplot as plt

n = 20

if __name__ == '__main__':
    ein_sum, eout_sum = [], []
    for t in range(1000):
        x = np.random.uniform(-1, 1, n)
        y = np.sign(x)
        y = [i if random.random() >= 0.2 else -i for i in y]
        ein = 7122
        eout = 0
        for s in [1, -1]:
            for theta in [-1.1] + list(x + 1e-15):
                e = ((s * np.sign(x - theta)) != y).mean()
                if ein > e:
                    ein = e
                    eout = 0.5 + 0.3 * s * (abs(theta) - 1)
        ein_sum.append(ein)
        eout_sum.append(eout)
    print(np.mean(ein_sum), np.mean(eout_sum))
    plt.scatter(ein_sum, eout_sum)
    plt.xlabel('Ein')
    plt.ylabel('Eout')
    plt.show()


