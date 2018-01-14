import numpy as np
import matplotlib.pyplot as plt

def gradient(w, x, y):
    e = np.exp(np.dot(x, w) * y)
    return ((1 / (1 + e) * -y * x)).mean(0)[:,np.newaxis]

def gradient_descent(x_train, y_train, x_test, y_test, sgd, eta):
    w = np.zeros((x_train.shape[-1], 1))
    ein, eout = [], []
    for t in range(2000):
        if sgd:
            grad = gradient(w, x_train[t%1000][np.newaxis,:], y_train[t%1000][np.newaxis,:])
        else:
            grad = gradient(w, x_train, y_train)
        w -= eta * grad
        ein.append((np.sign(np.dot(x_train, w)) != y_train).mean())
        eout.append((np.sign(np.dot(x_test, w)) != y_test).mean())
    return ein, eout

if __name__ == '__main__':
    train_data = np.loadtxt('hw3_train.dat')
    test_data = np.loadtxt('hw3_test.dat')
    x_train, y_train = train_data[:,:-1], train_data[:,-1,np.newaxis]
    x_test, y_test = test_data[:,:-1], test_data[:,-1,np.newaxis]

    sgd_in, sgd_out = gradient_descent(x_train, y_train, x_test, y_test, sgd=True, eta=0.01)
    gd_in, gd_out = gradient_descent(x_train, y_train, x_test, y_test, sgd=False, eta=0.01)
    
    plt.plot(gd_in, label='GD')
    plt.plot(sgd_in, label='SGD')
    plt.title('lr = 0.01')
    plt.xlabel('t')
    plt.ylabel('Ein')
    plt.legend()
    plt.show()
    
    plt.plot(gd_out, label='GD')
    plt.plot(sgd_out, label='SGD')
    plt.title('lr = 0.01')
    plt.xlabel('t')
    plt.ylabel('Eout')
    plt.legend()
    plt.show()
    
    sgd_in, sgd_out = gradient_descent(x_train, y_train, x_test, y_test, sgd=True, eta=0.001)
    gd_in, gd_out = gradient_descent(x_train, y_train, x_test, y_test, sgd=False, eta=0.001)
    
    plt.plot(gd_in, label='GD')
    plt.plot(sgd_in, label='SGD')
    plt.title('lr = 0.001')
    plt.xlabel('t')
    plt.ylabel('Ein')
    plt.legend()
    plt.show()
    
    plt.plot(gd_out, label='GD')
    plt.plot(sgd_out, label='SGD')
    plt.title('lr = 0.001')
    plt.xlabel('t')
    plt.ylabel('Eout')
    plt.legend()
    plt.show()
