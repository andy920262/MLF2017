import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    train_data = np.loadtxt('hw4_train.dat')
    test_data = np.loadtxt('hw4_test.dat')
    x_train, y_train = np.hstack([np.ones((200, 1)), train_data[:,:-1]]), train_data[:,-1]
    x_test, y_test = np.hstack([np.ones((1000, 1)), test_data[:,:-1]]), test_data[:,-1]
    ein, eout = [], []
    LAMBDA = 1e2
    for i in range(13):
        w = np.linalg.inv(x_train.T @ x_train + LAMBDA * np.eye(3)) @ x_train.T @ y_train
        ein.append((np.sign(x_train @ w) != y_train).mean())
        eout.append((np.sign(x_test @ w) != y_test).mean())
        LAMBDA /= 10
    plt.plot(range(2, -11, -1), ein, label='Ein')
    plt.plot(range(2, -11, -1), eout, label='Eout')
    plt.xlim(2, -11) 
    plt.xlabel(r'$log_{10} \lambda$')
    plt.legend()
    plt.show()

    x_valid, y_valid = x_train[120:], y_train[120:]
    x_train, y_train = x_train[:120], y_train[:120]
    ein, ev, eout = [], [], []
    LAMBDA = 1e2
    for i in range(13):
        w = np.linalg.inv(x_train.T @ x_train + LAMBDA * np.eye(3)) @ x_train.T @ y_train
        ein.append((np.sign(x_train @ w) != y_train).mean())
        ev.append((np.sign(x_valid @ w) != y_valid).mean())
        eout.append((np.sign(x_test @ w) != y_test).mean())
        LAMBDA /= 10
    plt.plot(range(2, -11, -1), ein, label='Ein')
    plt.plot(range(2, -11, -1), ev, label='Eval')
    plt.plot(range(2, -11, -1), eout, label='Eout')
    plt.xlim(2, -11) 
    plt.xlabel(r'$log_{10} \lambda$')
    plt.legend()
    plt.show()



