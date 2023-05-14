# This file compute the features from the contents to train the model.
import re 
from util.FeaturesComputation import FeaturesComputation as FC
import os
import pickle 
import numpy as np
from tqdm import tqdm


def column_selector():
    # This function returns a list of integers representing column indices from 0 to 545. 
    return [i for i in range(0, 546)]

def get_features():
    # Initializes the paths for the training and testing datasets.
    print("Computing features for training dataset - - - > > > >")
    
    if not os.path.exists(os.getcwd()+"/datasets/computedFeatures/"):
        os.makedirs(os.getcwd()+"/datasets/computedFeatures/")

    path = os.getcwd() + "/datasets/EBG-obfuscation/"

    train_path = os.path.join(path + "train/")
    test_path = os.path.join(path + "test/")
    # Initializes empty lists to store features and labels for both the training and testing datasets.
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    # Train set
    for file in tqdm(os.listdir(train_path)):
        
        # For each file in the training dataset, it reads the file's contents, 
        # computes its features using an object of the FC class, and appends the 
        # features and the label to the respective lists.
        fc = FC()
        name = re.findall("[a-z]+", file)[0]
        with open(train_path+file, "r", encoding="utf8") as f:
            contents = f.read()
            fc_content = fc.getFeatures(contents)

            train_X.append(fc_content)
            train_y.append(name)

        del fc

    # It stores the computed features and labels of the training dataset in pickled format.
    train_X_file = open(os.getcwd()+"/datasets/computedFeatures/train_X.pkl", "wb")
    pickle.dump(train_X, train_X_file)
    train_X_file.close()

    train_y_file = open(os.getcwd()+"/datasets/computedFeatures/train_y.pkl", "wb")
    pickle.dump(train_y, train_y_file)
    train_y_file.close()

    print("Computing features for testing dataset - - - > > > >")
    # Test set
    for file in tqdm(os.listdir(test_path)):
        fc = FC()
        name = re.findall("[a-z]+", file)[0]
        with open(test_path+file, "r", encoding="utf8") as f:
            contents = f.read()
            fc_content = fc.getFeatures(contents)
            
            test_X.append(fc_content)
            test_y.append(name)
        del fc

    # It stores the computed features and labels of the testing dataset in pickled format.
    test_X_file = open(os.getcwd()+"/datasets/computedFeatures/test_X.pkl", "wb")
    pickle.dump(test_X, test_X_file)
    test_X_file.close()

    test_y_file = open(os.getcwd()+"/datasets/computedFeatures/test_y.pkl", "wb")
    pickle.dump(test_y, test_y_file)
    test_y_file.close()
    

def load_data():
    # Load the train and test data from pickled files.
    train_X = pickle.load(open(os.path.join(os.getcwd()+"/datasets/computedFeatures/train_X.pkl"), "rb"))
    train_y = pickle.load(open(os.path.join(os.getcwd()+"/datasets/computedFeatures/train_y.pkl"), "rb"))
    test_X = pickle.load(open(os.path.join(os.getcwd()+"/datasets/computedFeatures/test_X.pkl"), "rb"))
    test_y = pickle.load(open(os.path.join(os.getcwd()+"/datasets/computedFeatures/test_y.pkl"), "rb"))
    # Return the train and test data as numpy matrices and numpy arrays.
    return np.asarray(train_X), train_y, np.asarray(test_X), test_y

# if __name__ == "__main__":
    # train_X, train_y, test_X, test_y = get_data()
    # print(train_X[0], train_y[0], test_X[0], test_y[0])
    # load_data()