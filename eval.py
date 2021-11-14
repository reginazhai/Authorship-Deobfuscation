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

import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    if(tensor_input.shape[1] > 512):
        return False
    else:
        loss=model(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)

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

testInstancesFilename = "../../Data/BlogsAllObf/BlogX_test.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances = pickle.load(f)

testInstancesFilename = "../../Data/BlogsAllObf/X_test/X_test_obf_dspan300.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances1 = pickle.load(f)

#testInstancesFilename = "../../Data/BlogsAllObf/X_test/X_test_obf_snpan300.pickle"
#with open(testInstancesFilename, 'rb') as f:
#        testInstances2 = pickle.load(f)

testInstancesFilename = "../../Data/BlogsAllObf/X_test_WP_300.pickle"
with open(testInstancesFilename, 'rb') as f:
        testInstances3 = pickle.load(f)

testInstance_total = testInstances + testInstances1 + testInstances3
print("Test Instances Length : ", len(testInstance_total))

incorrectCount = 0
correctCount = 0
total_author = {}
total_document = []
total_original = []
total_predicted = []
total_complexity = []
total_writeprint = []

for documentNumber in range(len(testInstance_total)):
        filePath, filename, authorId, author, inputText = testInstance_total[documentNumber]
        print("Document Name : ", filename)
        print("Document Number : ", documentNumber)
        #if(authorId != 14):
            #continue

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
        total_writeprint.append(writeprintsStatic.calculateFeatures(inputText))
        clf.getLabelAndProbabilities(originalDocument)
        print("Predicted author:", originalDocument.documentAuthor)
        total_original.append(originalDocument.documentAuthor)
        print("Original author:", authorId)
        total_predicted.append(authorId)
        
        excess_list = ['9-3403444.txt','9-1593902.txt','DS84-1474573','DS9-1593902']
        token_exceed = False
        #from nltk import sent_tokenize
        #sentenceText = sent_tokenize(inputText)
        #total_score = 0
        #print(sentenceText)
        #for sentence in sentenceText:
        #    cur_score = score(sentence)
        #    if (cur_score):
        #        print(sentence)
        #        print(cur_score)
        #        print("-------------------------")
        #        total_score += cur_score
        #    else:
        #        token_exceed = True
        #        break
        #if (not token_exceed):
        #    total_complexity.append(total_score/len(sentenceText))
        #    print("Average score:", total_score/len(sentenceText))
        #else:
        #    total_complexity.append("EXCEED")

        if originalDocument.documentAuthor != authorId:        
                incorrectCount += 1
                print(filename + " Classified InCorrectly")

        else:
                correctCount +=1
                print(filename + " Classified Correctly")
                
        #print("Predicted Proba:", originalDocument.documentAuthorProbabilites)
        print("--------------------------")

print("Incorrectly Classified:", incorrectCount)
print("Correctly Classified:", correctCount)
print("Accuracy:", correctCount/(correctCount+incorrectCount))
df = pd.DataFrame(total_writeprint)
files = {"FileName":total_document, "Predicted":total_original, "Original": total_predicted}
df = pd.DataFrame(files, columns = ["FileName", "Original", "Predicted"])
#df["Filename"] = total_document
df.to_excel("MutantX+DSPAN600.xlsx", sheet_name='Sheet1', engine='xlsxwriter')

