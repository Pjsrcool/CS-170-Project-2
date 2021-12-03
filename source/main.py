import random
import csv
import os
import pandas as pd
import numpy as np
import sys
import math
import time
from numba import jit

@jit
def EuclideanDistance(pointA, pointB, features_to_compare):
    sum = 0
    for i in range (0, len(features_to_compare)):
        sum += ((pointA[features_to_compare[i]] - pointB[features_to_compare[i]]) ** 2)
    return math.sqrt(sum)

@jit
def leave_one_out_cross_validaton_forward(data, current_set_of_features, feature_to_add):
    # return random.randint(0,10)

    number_correctly_classified = 0;
    features_to_compare = []
    features_to_compare.extend(current_set_of_features)
    features_to_compare.append(feature_to_add)
    # print("comparing features: " + str(features_to_compare))

    for i in range (0, len(data)):
        object_to_classify = data[i]
        label_object_to_classify = data[i][0]

        nearest_neighbor_distance = sys.maxsize
        nearest_neighbor_location = sys.maxsize
        nearest_neighbor_label = sys.maxsize

        for k in range (0,len(data)):
            if k != i:
                distance = EuclideanDistance(object_to_classify,data[k], features_to_compare)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    # print(number_correctly_classified)
    return number_correctly_classified / len(data)

@jit
def leave_one_out_cross_validaton_backward(data, current_set_of_features, feature_to_remove):
    # return random.randint(0,10)

    number_correctly_classified = 0;
    features_to_compare = []
    features_to_compare.extend(current_set_of_features)
    features_to_compare.remove(feature_to_remove)

    for i in range (0, len(data)):
        object_to_classify = data[i]
        label_object_to_classify = data[i][0]

        nearest_neighbor_distance = sys.maxsize
        nearest_neighbor_location = sys.maxsize
        nearest_neighbor_label = sys.maxsize

        for k in range (0,len(data)):
            if k != i:
                distance = EuclideanDistance(object_to_classify,data[k], features_to_compare)
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
    # print(number_correctly_classified)
    return number_correctly_classified / len(data)

@jit
def feature_search_forward_selection (data):
    current_set_of_features = [] # init empty set
    accuracy_after_each_add = []
    best_set_so_far = []
    actual_best_accuracy = 0

    for i in range (1, len(data[0])):
        
        print("on the " + str(i) + "th level of search tree")
        feature_to_add_at_this_level = None;
        best_so_far_accuracy = 0;
        for k in range (1, len(data[0])):
            if not (k in current_set_of_features):
                print("consider adding the feature " + str(k))
                accuracy = leave_one_out_cross_validaton_forward(data, current_set_of_features, k)
                # print(accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if (feature_to_add_at_this_level != None):
            current_set_of_features.append(feature_to_add_at_this_level)
            accuracy_after_each_add.append(best_so_far_accuracy)
        print("on level " + str(i) + " we added features " + str(feature_to_add_at_this_level))
        print(str(current_set_of_features) + " accuracy: " + str(best_so_far_accuracy))
        print()

        if (best_so_far_accuracy > actual_best_accuracy):
            best_set_so_far = []
            best_set_so_far.extend(current_set_of_features)
            actual_best_accuracy = best_so_far_accuracy

    print("best set is: " + str(best_set_so_far) + " with accuracy: " + str(actual_best_accuracy))
    return current_set_of_features, accuracy_after_each_add

@jit
def feature_search_backward_elimination(data):
    current_set_of_features = [] # init empty set
    accuracy_after_each_elimination = []
    best_set_so_far = []
    set_at_each_level = []
    actual_best_accuracy = 0

    # initialize current set of features to a list of all the features
    for i in range (1, len(data[0])):
        current_set_of_features.append(i)
    
    for i in range (1, len(data[0])):
        print("on the " + str(i) + "th level of search tree")
        feature_to_remove_at_this_level = None;
        best_so_far_accuracy = 0;
        for k in range (1, len(data[0])):
            if k in current_set_of_features:
                print("consider removing the feature " + str(k))
                accuracy = leave_one_out_cross_validaton_backward(data, current_set_of_features, k)
                # print(accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k
        
        if (feature_to_remove_at_this_level != None):
            current_set_of_features.remove(feature_to_remove_at_this_level)
            accuracy_after_each_elimination.append(best_so_far_accuracy)
        print("on level " + str(i) + " we removed features " + str(feature_to_remove_at_this_level))
        print(str(current_set_of_features) + " accuracy: " + str(best_so_far_accuracy))
        print()

        if (best_so_far_accuracy > actual_best_accuracy):
            best_set_so_far = []
            best_set_so_far.extend(current_set_of_features)
            set_at_each_level.append(best_set_so_far[0:len(best_set_so_far)])
            actual_best_accuracy = best_so_far_accuracy

    print("best set is: " + str(best_set_so_far) + " with accuracy: " + str(actual_best_accuracy))
    return set_at_each_level, accuracy_after_each_elimination

@jit
def main():
    os.chdir("data")

    # here, we prompt the user to select whether to use the small or large data set
    dataSet = input("Enter 1 for small data set.\nEnter 2 for large data set.\n--> ")
    size = ""
    if (dataSet == '1'):
        # my small set is 91
        size = "Small"
        setNumber = 91
        # setNumber = 86
    elif (dataSet == '2'):
        # big is 57
        size = "LARGE"
        setNumber = 57
    

    # open the file and read from it
    Data =  open("Ver_2_CS170_Fall_2021_" + size + "_data__" + str(setNumber) + ".txt", 'r')
    data = pd.read_csv(Data, sep="\s+", dtype=float, quoting=csv.QUOTE_NONNUMERIC)
    dataList = data.values.tolist()

    searchType = input ("Enter 1 for Forward Selection.\nEnter 2 for Backward Elimination.\n--> ")
    print()

    # begin calculation
    start_time = time.time()

    # find default rate
    class_1 = 0
    for row in dataList:
        if row[0] == 1:
            class_1 += 1
    if class_1 >= len(dataList):
        default_rate = class_1 / len(dataList)
    else:
        default_rate = (len(dataList) - class_1) / len(dataList)
    print("the default rate is (empty set) is " + str(default_rate))
    print()

    # perform feature search
    if searchType == '1':
        features, accuracy = feature_search_forward_selection(dataList)
    elif searchType == '2':
        features, accuracy = feature_search_backward_elimination(dataList)

    end_time = time.time()
    print()

    # print results
    if searchType == '1':
        print(str([]) + " --> accuracy " + str(default_rate))
        i = 1
        for a in accuracy:
            print(str(features[0:i]) + " --> accuracy " + str(a))
            i += 1
    elif searchType == '2':
        i = 0
        for a,f in zip(accuracy,features):
            print(str(f) + " --> accuracy " + str(a))
            i += 1
        print(str([]) + " --> accuracy " + str(default_rate))

    print("runtime: %s seconds" % (end_time - start_time))

main()