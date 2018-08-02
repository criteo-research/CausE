from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import math as mt
import pandas as pd
from scipy.sparse import csr_matrix

def load_movielens(dataset_location):
    """Load the movielens 100K dataset from disk"""

    split_characters = "\t"
    userid = []
    productid = []
    rating = []

    with open(dataset_location, "r") as dataset_file:
        for line in dataset_file:
            f = line.rstrip('\r\n').split(split_characters)
            userid.append(int(f[0]))
            productid.append(int(f[1]))
            rating.append(int(f[2]))

    # Make the sequence being at zero and contain no missing IDs
    _, userid = np.unique(np.asarray(userid), return_inverse=1)
    _, productid = np.unique(np.asarray(productid), return_inverse=1)

    return userid, productid, rating

def load_movielens10M(dataset_location):
    """Load the movielens dataset from disk"""

    split_characters = "::"
    userid = []
    productid = []
    rating = []

    with open(dataset_location, "r") as dataset_file:
        for line in dataset_file:
            f = line.rstrip('\r\n').split(split_characters)
            userid.append(int(f[0]))
            productid.append(int(f[1]))
            rating.append(float(f[2]))

        # Make the sequence being at zero and contain no missing IDs
    _, userid = np.unique(np.asarray(userid), return_inverse=1)
    _, productid = np.unique(np.asarray(productid), return_inverse=1)

    print("Dataset loaded")

    return userid, productid, rating

def create_movielens_userproduct_matrix(userid, productid, rating):
    """Create the matrix from the movielens dataset"""

    model = "expomf"
    ratings_threshold = 5

    num_unique_users = int(np.amax(userid)+1)
    num_unique_products = int(np.amax(productid)+1)

    print(num_unique_users)
    print(num_unique_products)

    view_matrix = np.zeros(shape=(num_unique_users, num_unique_products))
    view_matrix[:] = np.NAN

    for i in range(userid.shape[0]):

        if rating[i] < ratings_threshold:
            view_matrix[userid[i], productid[i]] = 0

        elif rating[i] >= ratings_threshold:
            view_matrix[userid[i], productid[i]] = 1

        #return view_matrix
        np.savetxt('ML_view.txt', view_matrix)

    print("Data Saved")

def load_netflix_dataset(dataset_location):

    df1 = pd.read_csv(dataset_location, header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1]) 
    df2 = pd.read_csv('combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    df3 = pd.read_csv('combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
    df4 = pd.read_csv('combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])

    df = df1
    df = df1.append(df2)
    df = df.append(df3)
    df = df.append(df4)

    df.index = np.arange(0, len(df))

    print("Netflix data loaded into Pandas")

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i,j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1,i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    # numpy approach
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
    movie_np = np.append(movie_np, last_record)

    # remove those Movie ID rows
    df = df[pd.notnull(df['Rating'])]

    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    productid = df['Movie_Id'].values
    userid = df['Cust_Id'].values
    ratings = df['Rating'].values

    _, userid = np.unique(userid, return_inverse=1)
    _, productid = np.unique(np.asarray(productid), return_inverse=1)

    return userid, productid, ratings

if __name__ == "__main__":

    #userid, productid, rating = load_movielens("/Users/s.bonner/Downloads/ml-100k/u.data")
    userid, productid, rating = load_movielens10M("/home/ubuntu/ml-10M100K/ratings.dat")
    create_movielens_userproduct_matrix(userid, productid, rating)