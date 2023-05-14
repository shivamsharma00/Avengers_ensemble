'''
This file calls the ensemble architecture and save the trained model.
'''
import os 
import sys
print(os.getcwd())
sys.path.append(os.getcwd())

from util.util import get_features, load_data, column_selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from mlxtend.feature_selection import ColumnSelector
import random
from sklearn.preprocessing import StandardScaler
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import pickle
import numpy as np

TOTAL_CLFS = 20

def pipeHiddenClf(features):
    # Function to create pipelines 
    classifiers = [make_pipeline(
        ColumnSelector(cols=random.sample(range(features), 400)),
        SVC(kernel='linear', probability=True, random_state=0),
    ) for _ in range(TOTAL_CLFS)]
    return classifiers

def ensemble1(features):
    return make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        EnsembleVoteClassifier(pipeHiddenClf(features), voting='soft', use_clones=False)
    )

if __name__ == '__main__':

    print("Loading dataset - - - > > > >")
    
    if not os.path.exists(os.path.join(os.getcwd()+"/datasets/computedFeatures/")):
        get_features()
    X_train, y_train, X_test, y_test = load_data()

    columns = ColumnSelector(cols=column_selector())  # to select specific columns from the dataset
    variance_thresh = VarianceThreshold()   # Feature selector that remove all low-variance features

    totalFeatuers = len(variance_thresh.fit_transform(columns.fit_transform(X_train))[0])
    print("total features - ", totalFeatuers)
    ensemble = ensemble1(totalFeatuers)
    ensemble = make_pipeline(columns, variance_thresh, ensemble)
    
    print("Starting Training - - - > > > >")
    ensemble.fit(X_train, y_train)
    
    if not os.path.exists(os.path.join(os.getcwd()+"../trainedModels/EBG" + "/" )):
        os.makedirs(os.path.join(os.getcwd()+"../trainedModels/EBG/" + "/"))

    filename = os.path.join(os.getcwd()+"../trainedModels/EBG" + "/" + 'ebg_model.sav')
    pickle.dump(ensemble, open(filename, 'wb'))

    print("Accuracy Score")
    print(accuracy_score(y_test, ensemble.predict(X_test)))

    # if not os.path.exists("../trainedModels/featureSet" + "/" ):
    #     os.makedirs("../trainedModels/featureSet" + "/")
