import numpy as np
from tqdm import tqdm

## 3. Matrix Factorization
def matrix_factorization(x_train, x_test, k=50, lamda=0.01, alpha=1e-4, epoch=100):
    indicator_train = x_train > 0
    indicator_test = x_test > 0
    num_test = np.sum(indicator_test)

    np.random.seed(2022)
    U = np.random.rand(10000, k) * 0.1
    V = np.random.rand(10000, k) * 0.1
    J = np.zeros((epoch))
    RMSE = np.zeros((epoch))

    for i in tqdm(range(epoch)):
        dU = np.dot(indicator_train * ( np.dot(U, V.T) - x_train), V) + 2 * lamda * U
        dV = np.dot(np.transpose(indicator_train * (np.dot(U, V.T) - x_train)), U) + 2 * lamda * V
        U = U - (alpha * dU)
        V = V - (alpha * dV)
        J[i] = 1/2 * pow(np.linalg.norm(indicator_train * (x_train - np.dot(U, V.T))),2) + \
                lamda * pow(np.linalg.norm(U),2) + \
                lamda * pow(np.linalg.norm(V),2)
        predict_score = np.dot(U, V.T)
        RMSE[i] = np.sqrt(np.sum(pow((predict_score * indicator_test-x_test),2))/len(num_test))

    return U, V, J, RMSE