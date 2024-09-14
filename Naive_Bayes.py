import math
from scipy import stats
import numpy as np
import csv

def main(train_filename, test_filename):
    train_file = csv.DictReader(open(f'{train_filename}'))
    rows_train = list(train_file)

    #NETFLIX QUESTION: Adjusts data to remove demographic data as instructed
    for row in rows_train:
        del row['Demographic']

    #Creates original count dictionaries
    y0 = {}
    y1 = {}
    trials = 0
    for elem in rows_train[0]:
        y0[elem] = 0
        y1[elem] = 0

    #Counts all instances of X_i=1 and adds to corresponding dictionary for either y=0, y=1
    for row in rows_train:
        trials += 1
        for elem in row:
            if row['Label'] == '0' and row[elem] == '1':
                y0[elem] += 1
            if row['Label'] == '1' and row[elem] == '1':
                y1[elem] += 1
    y0['Label'] = trials - y1['Label']

    #Creates distributions of P(X_i = 1 | y=0,1) using Maximum A Posteriori
    p_x_y0 = MAP_p(y0)
    p_x_y0['Prior'] = (y0['Label'] + 1) / (trials + 2)
    p_x_y1 = MAP_p(y1)
    p_x_y1['Prior'] = (y1['Label'] + 1) / (trials + 2)

    test_file = csv.DictReader(open(f'{test_filename}'))
    rows_test = list(test_file)

    #NETFLIX QUESTION: removes demographic info from testing data
    for row in rows_test:
        del row['Demographic']

    #Creates variables to keep track of classification stats. Runs through all given test rows,
    # performs naive bayes on the row, and compares algorithms classification to true value and
    # updates stats
    y0_correct = 0
    y0_total = 0
    y1_correct = 0
    y1_total = 0
    for row in rows_test:
        log_p_y0 = naive_bayes(p_x_y0, row)
        log_p_y1 = naive_bayes(p_x_y1, row)
        if row['Label'] == '0':
            y0_total += 1
            if log_p_y0 > log_p_y1:
                y0_correct += 1
        else:
            y1_total += 1
            if log_p_y1 > log_p_y0:
                y1_correct += 1

    #Printing accuracy measures
    print(f'Class 0: tested {y0_total}, correctly classified {y0_correct}')
    print(f'Class 1: tested {y1_total}, correctly classified {y1_correct}')
    print(f'Overall: tested {y0_total+y1_total}, correctly classified {y0_correct+y1_correct}')
    print(f'Accuracy = {(y0_correct+y1_correct) / (y0_total+y1_total)}')

    #Additional info for Netflix question
    print('P(Y=1) = ',p_x_y1['Prior'])
    print(f'P(X_i=1 | Y=1) {p_x_y1}')
    print(influential_movies(p_x_y0, p_x_y1))

#NETFLIX QUESTION: calculates influence scores for movies using the given formula
def influential_movies(y0_dist, y1_dist):
    influences = {}
    for elem in y0_dist:
        influences[elem] = 0
        influences[elem] = ((y1_dist[elem]) / (y1_dist[elem]*y1_dist['Prior'] + (y0_dist[elem])*(y0_dist['Prior']))) / ((1 - y1_dist[elem]) / ((1 - y1_dist[elem])*y1_dist['Prior'] + (1 - y0_dist[elem])*(y0_dist['Prior'])))

    # Sort the dictionary items based on values in descending order
    sorted_influences = sorted(influences.items(), key=lambda x: x[1], reverse=True)
    return(sorted_influences)

#Takes in the MAP distribution of probabilities for X_i=1 | Y=y and uses to run naive bayes on a single test row
def naive_bayes(dist, test):
    posterior = math.log(dist['Prior'])
    for elem in test:
        if elem != 'Label':
            if test[elem] == '1':
                posterior += math.log(dist[elem])
            else:
                posterior += math.log(1 - dist[elem])
    return posterior

#Calculates Maximum A Posteriori with Laplace Estimators for all X_i=1 | Y=y counts
def MAP_p(counts):
    new = {}
    for elem in counts:
        if elem != 'Label':
            new[elem] = (counts[elem] + 1) / (counts['Label'] + 2)
    return new

main('netflix-train.csv', 'netflix-test.csv')