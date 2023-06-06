from data import load_data, data_preprocessing_coo_matrix
from CF import get_similarity_matrix, user_based_CF_matrix, get_similarity_matrix_baseline, user_based_CF_baseline
from MF import matrix_factorization
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

users, train_data_new, test_data_new, movie_info = load_data()
x_train = data_preprocessing_coo_matrix(train_data_new)
x_test = data_preprocessing_coo_matrix(test_data_new)

user_similarity = get_similarity_matrix(x_train)
x_pred, RMSE = user_based_CF_matrix(user_similarity, x_train, x_test)
print("RMSE of CF (Matrix Form):", RMSE)

user_ids = test_data_new.new_user_id.values
item_ids = test_data_new.movie_id.values - 1
y_true = test_data_new.rating.values
user_similarity, mu, bx, by = get_similarity_matrix_baseline(x_train)
predictions = []
test_sample = len(y_true)
for i in tqdm(range(test_sample)):
    predictions.append(user_based_CF_baseline(user_similarity, x_train, mu, bx, by, 
                                          user_ids[i], item_ids[i], 3))
RMSE = np.sqrt(mean_squared_error(y_true, predictions))
print('RMSE of CF with Decomposition:', RMSE)
for k in [10, 50, 100]:
    for lamda in [1, 0.1, 0.01, 0.001]:
        U, V, J, RMSE = matrix_factorization(x_train, x_test, k, lamda, alpha=1e-4, epoch=100)
        np.savez("a-4_k{}_l{}.npz".format(k, lamda), J=J, RMSE=RMSE, U=U, V=V)

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['figure.figsize'] = (12, 8) 
plt.style.use("bmh")
plt.rcParams['axes.unicode_minus'] = False
r = np.load("a-4_k50_l0.01.npz")
print('RMSE of MF:', r['RMSE'])
plt.plot(range(len(r['RMSE'])), r['RMSE'])
plt.xlabel('Epoch', size=20)
plt.ylabel('RMSE', size=20)
plt.title(r'RMSE on $X_{\mathrm{test}}$', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('RMSE.pdf', bbox_inches = 'tight')
plt.plot(range(len(r['J'])), r['J'])
plt.xlabel('Epoch', size=20)
plt.ylabel('Loss Fucntion', size=20)
plt.title(r'Loss Function $J$ on $X_{\mathrm{test}}$', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('loss.pdf', bbox_inches = 'tight')