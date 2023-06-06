import pandas as pd
import os
from scipy.sparse import coo_matrix


def load_data(data_prefix = './data'):

    users = pd.read_table(os.path.join(data_prefix, 'users.txt'), header=None)
    users.columns = ['user_id']

    train_data = pd.read_table(os.path.join(data_prefix, 'netflix_train.txt'), sep=' ', header=None)
    train_data.columns = ['user_id', 'movie_id', 'rating', 'date']

    test_data = pd.read_table(os.path.join(data_prefix, 'netflix_test.txt'), sep=' ', header=None)
    test_data.columns = ['user_id', 'movie_id', 'rating', 'date']

    movie_titles = pd.read_table(os.path.join(data_prefix, 'movie_titles.txt'), header=None, encoding='ISO-8859-1')
    movie_id = [i.split(',')[0] for i in movie_titles[0]]
    movie_year = [i.split(',')[1] for i in movie_titles[0]]
    movie_name = [','.join(i.split(',')[2:]) for i in movie_titles[0]]

    movie_info = pd.DataFrame(columns=['movie_id', 'movie_year', 'movie_name'])
    movie_info['movie_id'] = movie_id
    movie_info['movie_year'] = movie_year
    movie_info['movie_name'] = movie_name
    
    # show data
    print(users.head())
    print(train_data.head())
    print(test_data.head())
    print(movie_info.head())
    
    users['new_user_id'] = range(len(users))
    train_data_new = pd.merge(train_data, users)
    test_data_new = pd.merge(test_data, users)

    return users, train_data_new, test_data_new, movie_info


def data_preprocessing_coo_matrix(data):
    num_movie = data.movie_id.max() 
    num_user = data.new_user_id.max() + 1

    row = data.new_user_id.values 
    column = data.movie_id.values - 1
    transform_data = data.rating.values

    sparse_matrix = coo_matrix((transform_data, (row, column)), shape=(num_user, num_movie))
    x = sparse_matrix.toarray()

    return x


def data_preprocessing_pandas_pivot(data):
    x = data.pivot(index='new_user_id', columns='movie_id', values='rating')
    return x.fillna(0).values