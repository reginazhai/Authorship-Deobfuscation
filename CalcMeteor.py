import random
import math
from operator import itemgetter
import copy
import os
import re
import numpy as np
import pandas as pd
import xlsxwriter
import time
import io
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import writeprintsStatic

import torch
from pycocoevalcap.meteor.meteor import Meteor

def calculateFitness(originalDocument, changedDocument):
    meteor = Meteor()
    if len(originalDocument.split()) < 3000:
        meteor_score = meteor._score(''.join(str(e) + " " for e in changedDocument.split()), [''.join(str(e) + " " for e in originalDocument.split())])
    else:
        meteor_score = meteor._score(''.join(str(e) + " " for e in changedDocument.split()[:3000]), [''.join(str(e) + " " for e in originalDocument.split()[:3000])])
    return meteor_score

classifierType = 'ml'
authorstoKeep = 15
datasetName = 'BlogsAll'

testInstancesFilename = "Data/X_test/BlogX_test.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances = pickle.load(f)
# For Mutant-X
#testInstancesFilename = "Data/X_test/X_test_WP_300.pickle"
#with open(testInstancesFilename, 'rb') as f:
#        testInstances2 = pickle.load(f)

testInstancesFilename = "Data/X_test/X_test_obf_dspan300.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances3 = pickle.load(f)

print("Test Instances Length : ", len(testInstances))
# For Mutant-X
#print("Test Instances Length : ", len(testInstances2))
print("Test Instances3 Length : ", len(testInstances3))

total = {}

for documentNumber in range(len(testInstances)):
        filePath, filename, authorId, author, inputText = testInstances[documentNumber]
        total[filename] = {'Original':[authorId,inputText]}
# For Mutant-X
#for documentNumber in range(len(testInstances2)):
#        filePath, filename, authorId, author, inputText = testInstances2[documentNumber]
#        total[filename[11:] + '.txt']['Obfuscated'] = [authorId, inputText]

for documentNumber in range(len(testInstances3)):
        filePath, filename, authorId, author, inputText = testInstances3[documentNumber]
        total[filename[2:] + '.txt']['Obfuscated'] = [authorId, inputText]

for docname,text_dic in total.items():
        orig_text = text_dic['Original'][1]
        changed_text = text_dic['Obfuscated'][1]
        score = calculateFitness(orig_text,changed_text)
        print(docname,score)


