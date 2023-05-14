'''
This file contains code for avengers ensemble.

'''
# importing libraries
import pickle
import sys
from sklearn.svm import SVC
import os 
import io
import numpy as np
from util.FeaturesComputation import FeaturesComputation as FC
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
# Setting the parameter

try:
    database_name = str(sys.argv[0])
    required_authors = str(sys.argv[1])
except:
    database_name, required_authors= 'amt', 5

class Classifier:

    def __init__(self, authorName, index):

        self.authorName = authorName
        self.index = index
        self.classifier = None 

    def load_Classifier(self):

        classifier_path = str(PROJECT_ROOT)+"/trainedModels/ebg_model.sav"
        self.classifier = pickle.load(open(classifier_path, 'rb'))

    def get_label_and_probabilities(self, inputText):

        temp_file = open(str(self.index) + "temp_text", "w")
        temp_file.write(inputText.documentText)
        temp_file.close()

        inText = io.open(str(self.index) + "temp_text", "r", errors="ignore").readlines()
        inText = ''.join(str(e) +"" for e in inText)

        features = np.asarray([FC.getFeatures(inText)])

        prediction_prob = list(self.classifier.predict_proba(features)[0])

        clsases = self.classifier.classes_
        inputText.docAuthorProb = {key:value for (key, value) in enumerate(prediction_prob)}
        inputText.docAuthor = self.classifier.predict(features)[0] 
        os.remove(str(self.index) + "temp_text")

    def callClasses(self):
        return self.classifier.classes_
    
