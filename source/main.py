import random
import csv
import os
import pandas as pd
import numpy as np

def leave_one_out_cross_validaton(data, current_set_of_features, feature_to_add):
    return random.randint(0,10)

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

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        
        if (feature_to_add_at_this_level != None):
            current_set_of_features.append(feature_to_add_at_this_level)
        print("on level " + str(i + 1) + " we added features ")
        print(feature_to_add_at_this_level)
        print()

    print(current_set_of_features)



print("start")
# feature_search_demo([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

def main():
    os.chdir("data")
    with open("../data/CS170_Fall_2021_SMALL_data__91.txt", 'r') as smallData:
        # data = csv.reader(smallData, delimiter=' ',quoting=csv.QUOTE_NONNUMERIC)
        data = pd.read_csv(smallData, delimiter='\t', dtype=float)
        dataList = list(data)
        print(dataList[0])
        # print(len(dataList[0]))
        # feature_search_demo(dataList)

main()