import firebase_admin
from firebase_admin import firestore, credentials
import pandas as pd
import numpy as np
from scipy.linalg import lu as lu_factorization
from flask import Flask, jsonify
from tabulate import tabulate
import requests
from flask_restful import Api, Resource
from random2 import randint
import gunicorn

# R is the real matrix we are trying to estimate
# P and Q are the estimates which when multiplied give out an estimate of R
# K is the number of features, remember user-feature and feature-item = user-item ? features are represented by K here.
# steps are the number of times the experiment is repeated so that within which we get the best estimates for R


def matrix_factorization(R, P, Q, K, steps=1000, learning_rate=0.0002, regularization_parameter=0.02):
    Q = Q.T
    first_matrix = []
    # FIRST TASK IS ADJUSTING THE ENTRIES GIVEN BY P AND Q MATRIES BY A RANDOM FUNCTION IN NUMPY LIBRARY
    # did put steps just to remove the unused variable error, but any name could be placed in that place, good thing it doesn't affect a thing.
    for steps in range(steps):
        for i in range(len(R)):  # gives the number of rows
            for j in range(len(R[i])):  # gives the number of columns
                # using only the available entries, we placed 0's to just enable factorization.
                if R[i][j] > 0:
                    # first_matrix.__add__(R[i][j])
                    # getting the error for the respective element in R, because it is used in adjusting the P and Q elements
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + learning_rate * \
                            (2 * eij * Q[k][j] -
                             regularization_parameter * P[i][k])
                        Q[k][j] = Q[k][j] + learning_rate * \
                            (2 * eij * P[i][k] -
                             regularization_parameter * Q[k][j])

    # SECOND TASK IS FINDING THE TOTAL ERROR FOR BOTH PREDICTED VALUES
    # rememebr this error e is the squared error.
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    # returning to the original error or loss function so that we get and check the total error obtained
                    for k in range(K):
                        # the original code had regularization_parameter /2 , but i removed that division because i could find the reason behind it
                        e = e + (regularization_parameter) * \
                            (pow(P[i][k], 2) + pow(Q[k][j], 2))

    # THIRD TASK IS CHECKING IF THE TOTAL ERROR IS LESS THAN THE AGREED VALUE, IN ORDER TO SAVE TIME
        if e < 0.000005:
            break
    return P, Q.T


#getting zero and non-zero elements in the original matrix
def elements_in_the_original_matrix(matrix):
    nonzero_entries_rows = []
    nonzero_entries_columns = []
    zero_entries_rows = []
    zero_entries_columns = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                zero_entries_rows.append(i)
                zero_entries_columns.append(j)
            else:
                nonzero_entries_rows.append(i)
                nonzero_entries_columns.append(j)
    return nonzero_entries_rows, nonzero_entries_columns, zero_entries_rows, zero_entries_columns


#################################
# CLOUD FIRESTORE 
#Getting the ratings from firestore from the users
credential = credentials.Certificate("firebaseService.json")
firebase_admin.initialize_app(credential)
database = firestore.client()

ratings_matrix = []
snapshots = list(database.collection('Ratings').get())
for snapshot in snapshots:
    ratings_list = snapshot.to_dict()['ratings']
    ratings_array = np.array(ratings_list)
    if(len(ratings_matrix) == 0):
        ratings_matrix = ratings_array
    else:
        ratings_matrix = np.vstack([ratings_matrix, ratings_array])

# CALCULATION
R = np.array(ratings_matrix)
R = np.array(R)
N = len(R)
M = len(R[0])
K = 2
P = np.random.rand(N, K)
Q = np.random.rand(M, K)

new_P, new_Q = matrix_factorization(R, P, Q, K)
new_R = np.dot(new_P, new_Q.T)


# CONVERTING THE RESULTS TO INTERGERS
for row in range(len(new_R)):
    for column in range(len(new_R[0])):
        new_R[row][column] = round(new_R[row][column])

#converting the array to lists for each user & shipping back to firestore
def return_recommendations_to_firestore(matrix):
    recommendation_list = []
    for user in range(len(matrix)):
        recommendation_list.append(matrix[user].tolist())
        database.collection('Recommendations').document(f'{user + 1000}').set({'recommendations': matrix[user].tolist()})
    return recommendation_list


#compares values which had numbers in the original matrix only
def compare_predicted_vs_actual():
    table = [['Actual','Predicted','Error']]
    for index in range(len(nonzero_entries_rows)):
        row = nonzero_entries_rows[index]
        column = nonzero_entries_columns[index]
        actual = R[row][column]
        predicted = new_R[row][column]
        table.append([actual, predicted, round(actual - predicted, 2)])
    return table


nonzero_entries_rows, nonzero_entries_columns, zero_entries_rows, zero_entries_columns = elements_in_the_original_matrix(R)

#returns only values which were initially zeros
def only_recommendations(matrix):
    only_recommendations_dictioanry = {}
    i = 0
    for user in range(len(matrix)):
        only_recommendations_dictioanry[user + 1] = {}
        only_recommendations_dictioanry[user  + 1]['Recommendations'] = []
        movieIds = zero_entries_columns[i:i + zero_entries_rows.count(user)]
        only_recommendations_dictioanry[user  + 1]['MovieIds'] = movieIds
        for index in range(zero_entries_rows.count(user)):
            row = zero_entries_rows[index]
            column = zero_entries_columns[index]
            recommendation = new_R[row][column]
            only_recommendations_dictioanry[user + 1]['Recommendations'].append(recommendation)
        i += zero_entries_rows.count(user)
    return only_recommendations_dictioanry


recommendations = return_recommendations_to_firestore(new_R)
only_recommendations = only_recommendations(new_R)
table = tabulate(compare_predicted_vs_actual()[0:10], headers = 'firstrow', tablefmt='fancy_grid')

#################################
# THE API
api_website = Flask(__name__)

@api_website.route('/', methods=['GET'])
def homepage():
    return jsonify(results = only_recommendations)

if __name__ == "__main__":
    api_website.run()





#IMPORTANT NOTES
#the procfile like here with before : the file you wrote your code and after : the place where you run the flask app, for this case they are both on the same file
#requirements.txt to contain every module you've used in the code, it also has to include gunicorn, the module you use for deploying the app as a website
#follow the heroku starting guideline here: https://medium.com/unitechie/deploying-a-flask-application-to-heroku-b8814beaa954
#on updating the code on heroku just start with adding the files you have only updated with git add app.py requirements.txt for example. and proceed 
#make sure python runtime matches the ones supported by heroku