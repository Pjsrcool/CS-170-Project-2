import random
import csv
import os
import pandas as pd
import numpy as np
import sys
import math

def EuclideanDistance(pointA, pointB, features_to_compare):
    sum = 0
    for i in range (0, len(features_to_compare)):
        sum += ((pointA[features_to_compare[i]] - pointB[features_to_compare[i]]) ** 2)
    return math.sqrt(sum)

def leave_one_out_cross_validaton_forward(data, current_set_of_features, feature_to_add):
    # return random.randint(0,10)

    number_correctly_classified = 0;
    features_to_compare = []
    features_to_compare.extend(current_set_of_features)
    features_to_compare.append(feature_to_add)
    print("comparing features: " + str(features_to_compare))

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

def feature_search_forward_selection (data):
    current_set_of_features = [] # init empty set
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
                print(accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if (feature_to_add_at_this_level != None):
            current_set_of_features.append(feature_to_add_at_this_level)
        print("on level " + str(i) + " we added features " + str(feature_to_add_at_this_level))
        # print("on level " + str(i) + " we have features " + str(current_set_of_features))
        print(str(current_set_of_features) + " accuracy: " + str(best_so_far_accuracy))
        print()

        if (best_so_far_accuracy > actual_best_accuracy):
            best_set_so_far = []
            best_set_so_far.extend(current_set_of_features)
            actual_best_accuracy = best_so_far_accuracy

    # print(current_set_of_features)
    print("best set is: " + str(best_set_so_far) + " with accuracy: " + str(actual_best_accuracy))

def feature_search_backward_elimination(data):
    current_set_of_features = [] # init empty set
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
        print("on level " + str(i) + " we removed features " + str(feature_to_remove_at_this_level))
        # print("on level " + str(i) + " we have features " + str(current_set_of_features))
        print("with accuracy " + str(best_so_far_accuracy))
        print(current_set_of_features)
        print()

    print(current_set_of_features)

def main():
    os.chdir("data")
    # my small set is 91
    # big is 57

    # setNumber = 27

    dataSet = input("Enter 1 for small data set.\nEnter 2 for large data set.\n--> ")
    size = ""
    if (dataSet == '2'):
        size = "LARGE"
        setNumber = 57
    else:
        size = "Small"
        setNumber = 91


    with open("Ver_2_CS170_Fall_2021_" + size + "_data__" + str(setNumber) + ".txt", 'r') as Data:
        # data = csv.reader(smallData, delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        data = pd.read_csv(Data, sep="\s+", dtype=float, quoting=csv.QUOTE_NONNUMERIC)
        dataList = data.values.tolist()
        # print(dataList[0])
        # print(len(dataList[0]))

        # testAccuracy = leave_one_out_cross_validaton_forward(dataList, [7,4], 9)
        # print(testAccuracy)
        searchType = input ("Enter 1 for Forward Selection.\nEnter 2 for Backward Elimination.\n--> ")
        print()

        if (searchType == '1'):
            feature_search_forward_selection(dataList)
        else:
            feature_search_backward_elimination(dataList)

main()