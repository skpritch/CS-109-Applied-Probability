import math
from scipy import stats
import numpy as np
import csv

#Takes in files, performs regression, reports accuracy
def logistic_regression(train_filename, test_filename):
    #Loads csv training file and formats
    train_file = csv.reader(open(f'{train_filename}'))
    rows_train = list(train_file)
    format_data(rows_train)

    #Intializes given constants and parameter, gradient arrays
    learn_rate = 0.000001
    steps = 1000
    parameters = np.zeros(len(rows_train[0]) - 1)
    gradient = []

    #Gradient ascent steps
    for i in range(steps):
        gradient = np.zeros(len(rows_train[0]) - 1)
        for h in range(1, len(rows_train)):
            for j in range(len(gradient)):
                gradient[j] += float(rows_train[h][j]) * (float(rows_train[h][len(rows_train[h]) - 1]) - sigmoid(rows_train[h], parameters))
        for j in range(len(parameters)):
            parameters[j] += (learn_rate * gradient[j])

    #Loads csv testing file and formats
    test_file = csv.reader(open(f'{test_filename}'))
    rows_test = list(test_file)
    format_data(rows_test)

    #Intializing variables for accuracy stats, runs predictions on each row in testing file
    y0_correct = 0
    y0_total = 0
    y1_correct = 0
    y1_total = 0
    for i in range (1, len(rows_test)):
        p_y1 = sigmoid(rows_test[i], parameters)
        if rows_test[i][len(rows_test[i])  -1] == '1':
            y1_total += 1
            if p_y1 > 0.5:
                y1_correct += 1
        else:
            y0_total += 1
            if 1 - p_y1 > 0.5:
                y0_correct += 1

    #Printed Accuract Stats
    print(f'Class 0: tested {y0_total}, correctly classified {y0_correct}')
    print(f'Class 1: tested {y1_total}, correctly classified {y1_correct}')
    print(f'Overall: tested {y0_total + y1_total}, correctly classified {y0_correct + y1_correct}')
    print(f'Accuracy = {(y0_correct + y1_correct) / (y0_total + y1_total)}')

    #Calling functions for NETFLIX QUESTIONS
    #print(llikelihood(rows_train, np.zeros(len(parameters))))
    #print(llikelihood(rows_train, parameters))
    #print(predictors(parameters, rows_test[0]))

#Formats data files: removes demographic column, adds intercept
def format_data(data):
    if 'Demographic' in data[0]:
        loc = data[0].index('Demographic')
        for row in data:
            del row[loc]
    for row in data:
        row.insert(0, 1)

#Takes the dot product of observations and parameters and squashes using sigmoid
def sigmoid(observed, parameters):
    sum = 0
    for i in range (len(parameters)):
        sum += (float(observed[i]) * float(parameters[i]))
    return 1 / (1 + math.exp(-sum))

#NETFLIX QUESTION: Matches movie titles to parameter weights
def predictors(parameters, movies):
    predictors = {}
    for i in range (len(movies) - 1):
        predictors[f'{movies[i]}'] = parameters[i]
    sorted_predictors = sorted_influences = sorted(predictors.items(), key=lambda x: x[1], reverse=True)
    return sorted_predictors

#NETFLIX QUESTION: Calculates log likelihood of data given input parameters
def llikelihood(data, parameters):
    LL = 0
    for i in range(1, len(data)):
        y = float(data[i][len(data[i]) - 1])
        sig = float(sigmoid(data[i], parameters))
        LL += (y * math.log(sig) + (1 - y) * math.log(1-sig))
    return LL

logistic_regression('heart-train.csv', 'heart-test.csv')