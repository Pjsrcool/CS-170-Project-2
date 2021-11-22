import random
import csv
import os
import pandas as pd
import numpy as np
import sys
import math

def EuclideanDistance(pointA, pointB, features_to_compare):
    sum = 0
    for i in range (1, len(features_to_compare)):
        sum += ((pointA[features_to_compare[i]] - pointB[features_to_compare[i]]) ** 2)
    return math.sqrt(sum)

def leave_one_out_cross_validaton(data, current_set_of_features, feature_to_add):
    # return random.randint(0,10)

    number_correctly_classified = 0;
    features_to_compare = []
    features_to_compare.append(current_set_of_features)
    features_to_compare.append(feature_to_add)

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

def feature_search_demo (data):
    current_set_of_features = [] # init empty set

    for i in range (1, len(data[0])):
        
        print("on the " + str(i) + "th level of search tree")
        feature_to_add_at_this_level = None;
        best_so_far_accuracy = 0;
        for k in range (1, len(data[0])):
            if not (k in current_set_of_features):
                print("consider adding the feature " + str(k))
                accuracy = leave_one_out_cross_validaton(data, current_set_of_features, k)
                # print(accuracy)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if (feature_to_add_at_this_level != None):
            current_set_of_features.append(feature_to_add_at_this_level)
        print("on level " + str(i) + " we added features " + str(feature_to_add_at_this_level))
        # print("on level " + str(i) + " we have features " + str(current_set_of_features))
        print("with accuracy " + str(best_so_far_accuracy))
        print()

    print(current_set_of_features)



print("start")
# feature_search_demo([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

def main():
    os.chdir("data")
    with open("../data/CS170_Fall_2021_SMALL_data__91.txt", 'r') as smallData:
        # data = csv.reader(smallData, delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        data = pd.read_csv(smallData, sep="\s+", dtype=float, quoting=csv.QUOTE_NONNUMERIC)
        dataList = data.values.tolist()
        # print(dataList[0])
        # print(len(dataList[0]))
        feature_search_demo(dataList)

main()