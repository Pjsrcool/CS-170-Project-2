import random

def leave_one_out_cross_validaton(data, current_set_of_features, feature_to_add):
    return random.randint(0,10)

def feature_search_demo (data):
    current_set_of_features = [] # init empty set

    for i in range (0, len(data)):
        
        print("on the " + str(i + 1) + "th level of search tree")
        feature_to_add_at_this_level = None;
        best_so_far_accuracy = 0;
        for k in range (0, len(data)):
            if not (data[k] in current_set_of_features):
                print("consider adding the feature " + str(data[k]))
                accuracy = leave_one_out_cross_validaton(data, current_set_of_features, data[k])

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = data[k]
        
        if (feature_to_add_at_this_level != None):
            current_set_of_features.append(feature_to_add_at_this_level)
        print("on level " + str(i + 1) + " we added features ")
        print(feature_to_add_at_this_level)
        print()

    print(current_set_of_features)



print("start")
# tree = [
#     [],
#     [1,2,3,4],
#     [1,2],[1,3]
# ]
feature_search_demo([1,2,3,4])