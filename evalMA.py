import argparse
import random
import math
import gensim.models.keyedvectors as word2vec
from Document import Document
from Classifier import Classifier
from Mutant import Mutant_X
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

#dir_list = os.listdir('/home/rzhai/Desktop/M100_Obfuscated/')

classifierType = 'ml'
authorstoKeep = 15
datasetName = 'BlogsAll'

def getInformationOfInputDocument(documentPath):
    authorslabeler = LabelEncoder()
    authorslabeler.classes_ = np.load('Classes.npy')

    inputText = io.open(documentPath, "r", errors="ignore").readlines()
    inputText = ''.join(str(e) + "" for e in inputText)
 
    ## For SN-PAN16 && MutantX, change to -2
    authorName = (documentPath.split('/')[-3]).split('_')[0]
    authorLabel = authorslabeler.transform([authorName])[0]

    return (authorLabel, authorName, inputText)

testInstancesFilename = "Data/X_test/X_test_WP_300.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances = pickle.load(f)

print("Test Instances Length : ", len(testInstances))

incorrectCount = 0
correctCount = 0
total_author = {}
total_document = []
total_original = []
total_predicted = []

for documentNumber in range(len(testInstances)):
        filePath, filename, authorId, author, inputText = testInstances[documentNumber]
        print("Document Name : ", filename)
        print("Document Number : ", documentNumber)

        clf = Classifier(classifierType, authorstoKeep, datasetName, filename)
        clf.loadClassifier()
	#filePath = '/home/rzhai/Desktop/M100_Obfuscated/'+documentName + '/100/'
        #filePath = '/home/rzhai/Desktop/SN-PAN16_Obfuscated/' + documentName + '/'
        ## For Original Documents
	#filePath = '/home/rzhai/MutantX/MutantX/Data/datasetPickles/amt-10/X_test/'
	## For SN-PAN16
	#fileName = documentName + '.txt'
	
	## For DS-PAN17 && MutantX
        '''
	fileName = sorted(os.listdir(filePath))[-2]
	filePath = filePath + fileName
	filename = documentName
	authorId, author, inputText = getInformationOfInputDocument(filePath)
        '''
        if (filename=='SN84-1593902.txt'):
            continue
        total_document.append(filename)

        originalDocument = Document(inputText)
        clf.getLabelAndProbabilities(originalDocument)
        print("Original author:", originalDocument.documentAuthor)
        total_original.append(originalDocument.documentAuthor)
        print("Predicted author:", authorId)
        total_predicted.append(authorId)
        if originalDocument.documentAuthor != authorId:        
                incorrectCount += 1
                print("--------------------------")
                print(filename + " Classified InCorrectly")

        else:
                correctCount += 1
                print("--------------------------")
                print(filename + " Classified Correctly")

print("Incorrectly Classified:", incorrectCount)
print("Correctly Classified:", correctCount)
print("Accuracy:", correctCount/(correctCount+incorrectCount))

df = pd.DataFrame([total_document,total_original,total_predicted])
df.to_excel("test.xlsx", engine='xlsxwriter')


