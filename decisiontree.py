import numpy as np
import matplotlib.pyplot as plt



"""""""""
data set with 3 features 
"""""""""

x_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))


def compute_entropy(y):
    """
    Computes the purity of data for y=1 

    """
    entropy = 0.
    
    n=(y==1).sum()
    m=len(y)
    p1=n/m
    if p1==0.0 or p1==1.0:
        entropy=0.0
    else:
        entropy=-p1*np.log2(p1)-(1-p1)*np.log2(1-p1)
    
    return entropy




def split_dataset(x, node_indices, feature):
    """

    Split the data into left(y=1) and right(y=0) branches
    
    """
    
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if x[i,feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
        
    return left_indices, right_indices


all_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def compute_information_gain(x, y, node_indices, feature):
    
    """

    Compute the information gain from formula on splitting the node for given feature
    
    """    

    left_indices, right_indices = split_dataset(x, node_indices, feature)
    
    x_node, y_node = x[node_indices], y[node_indices]
    x_left, y_left = x[left_indices], y[left_indices]
    x_right, y_right = x[right_indices], y[right_indices]
    
    information_gain = 0.0
    
    hnode = compute_entropy(y_node)
    
    wleft = len(y_left) / len(y_node)
    hleft = compute_entropy(y_left)
    
    wright = len(y_right) / len(y_node)
    hright = compute_entropy(y_right)
    
    
    
    wavg = wleft * hleft + wright * hright
    information_gain=hnode-wavg
    
    return information_gain

info_gain0 = compute_information_gain(x_train, y_train, all_indices, feature=0)
info_gain1 = compute_information_gain(x_train, y_train, all_indices, feature=1)
info_gain2 = compute_information_gain(x_train, y_train, all_indices, feature=2)

print(info_gain0 , info_gain1 , info_gain2)




def best_split(x, y, node_indices):   

    """
    Get the best feature w min entropy for root out of all features

    """    
    
    num_features = x.shape[1]
    
    best_feature = -1
    max_info_gain=0
    infogain=np.zeros((num_features))
    for i in range(num_features):
        info=compute_information_gain(x, y, node_indices, i)
        if info>max_info_gain:
            max_info_gain=info
            best_feature=i
    
    return best_feature 


best_feature = best_split(x_train, y_train, all_indices)
print(best_feature)




tree = []

def decision_tree(x, y, node_indices, branch_name, max_depth, current_depth):
    """
    Combine all and form decision tree
    
    """ 

    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    best_feature = best_split(x, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    left_indices, right_indices = split_dataset(x, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    decision_tree(x, y, left_indices, "Left", max_depth, current_depth+1)
    decision_tree(x, y, right_indices, "Right", max_depth, current_depth+1)




"""
implement decision tree for our dataset depth from 0,1,2 feature index
"""


decision_tree(x_train, y_train, all_indices, "Root", max_depth=2, current_depth=0)