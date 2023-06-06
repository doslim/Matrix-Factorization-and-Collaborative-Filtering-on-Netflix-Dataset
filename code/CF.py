from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import mean_squared_error


## Get similarity matrix
def get_similarity_matrix(score_matrix):
    return cosine_similarity(score_matrix)

## Matrix form with valid users
def user_based_CF_matrix(similarity_matrix, score_matrix, test_matrix):
    indicator_test = test_matrix > 0
    indicator_train = score_matrix > 0
    x_pred = np.dot(similarity_matrix, score_matrix) / np.dot(similarity_matrix, indicator_train)
    RMSE = np.sqrt(mean_squared_error(x_pred[indicator_test], test_matrix[indicator_test]))
    return x_pred, RMSE

## Consider top-K users
### Do not consider valid users
def user_based_CF(similarity_matrix, score_matrix, user_id, item_id, k=5):
    '''
    user_based_CF: to generate recommendations based on similar users
    
    : parameters
    - similarity_matrix: numpy.ndarray, contains the similarity score between users
    - score_matrix: numpy.ndarray, known matrix that contains known rating scores
    - user_id: int, the user_id to query
    - item_id: int, the item_id to query
    - k: int, default 5, the number of similar users to be consider
    
    : return
    - rating: the predicted rating of the user the movie
    '''
    
    
    idx = np.argsort(similarity_matrix[user_id])[-k-1:-1] 
    sim = similarity_matrix[user_id, idx]
    score = score_matrix[idx, item_id]
    rating = np.sum(sim * score) / np.sum(sim)
    return rating


### Consider Valid Users
def user_based_CF_valid_user(similarity_matrix, score_matrix, user_id, item_id, k=5):
    '''
    user_based_CF: to generate recommendations based on similar users
    
    : parameters
    - similarity_matrix: numpy.ndarray, contains the similarity score between users
    - score_matrix: numpy.ndarray, known matrix that contains known rating scores
    - user_id: int, the user_id to query
    - item_id: int, the item_id to query
    - k: int, default 5, the number of similar users to be consider
    
    : return
    - rating: the predicted rating of the user the movie
    '''
    
    
    valid_similar_user = score_matrix[:,item_id].nonzero()[0]
    k = min(k, len(valid_similar_user))
    idx = np.argsort(similarity_matrix[user_id, valid_similar_user])[-k:] 
    idx = valid_similar_user[idx]
    sim = similarity_matrix[user_id, idx]
    score = score_matrix[idx, item_id]
    rating = np.sum(sim * score) / np.sum(sim)
    return rating


### Score decomposition
def get_similarity_matrix_baseline(score_matrix):
    x = score_matrix.astype(np.float64)
    mu = np.sum(x) / np.count_nonzero(x)
    bx = np.sum(x, axis=1) / np.count_nonzero(x, axis=1) - mu
    by = np.sum(x, axis=0) / np.count_nonzero(x, axis=0) - mu
    x[x==0] = 1000
    temp = x - (bx + mu).reshape(-1,1)
    temp[temp>10] = 0
    similarity_matrix = cosine_similarity(temp)

    return similarity_matrix, mu, bx, by


def user_based_CF_baseline(similarity_matrix, score_matrix, mu, bx, by, user_id, item_id, k=3):
    '''
    user_based_CF: to generate recommendations based on similar users
    
    : parameters
    - similarity_matrix: numpy.ndarray, contains the similarity score between users
    - score_matrix: numpy.ndarray, known matrix that contains known rating scores
    - user_id: int, the user_id to query
    - item_id: int, the item_id to query
    - k: int, default 3, the number of similar users to be consider
    
    : return
    - rating: the predicted rating of the user the movie
    '''
    
    valid_similar_user = score_matrix[:,item_id].nonzero()[0]
    k = min(k, len(valid_similar_user))
    idx = np.argsort(similarity_matrix[user_id, valid_similar_user])[-k:] 
    idx = valid_similar_user[idx]
    user_baseline = mu + bx[user_id] + by[item_id]
    similar_user_baseline = mu + bx[idx] + by[item_id]
    sim = similarity_matrix[user_id, idx]
    score = score_matrix[idx, item_id]
    if np.sum(sim) < 1e-5:
        rating = user_baseline
        return rating
    
    rating = user_baseline + np.sum(sim * (score - similar_user_baseline)) / np.sum(sim)
    return rating