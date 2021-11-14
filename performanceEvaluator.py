import os
import sys
import io
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import writeprintsStatic as ws
from pycocoevalcap.meteor.meteor import Meteor
import pandas as pd

meteor = Meteor()
resultsFolder = sys.argv[1]
attributionClassifier = "../../AuthorshipAttributionSystems/RFCWriteprintsStatic/trainedModels/" + str('amt') + '-' + str(10) + '/trained_model.sav'
clf = pickle.load(open(attributionClassifier, 'rb'))
authorslabeler = LabelEncoder()
authorslabeler.classes_ = np.load('classes.npy')

def getBestRun(inputlist):
    sorted_by_second = sorted(inputlist, key=lambda tup: tup[1])
    return sorted_by_second[-1]

def Obfuscated_or_Not(obfuscatedDocument, author):
    orig_label = authorslabeler.transform([author])[0]
    curr_label = clf.predict([ws.calculateFeatures(obfuscatedDocument)])[0]
    if orig_label != curr_label:
        return True
    else:
        return False

def getDirs(folder):
    return [f.path for f in os.scandir(folder) if f.is_dir() ]

def getMaxIterationNumberAvailable(filePath):
    iterations = []
    for (_,_,namess) in os.walk(filePath):
        iterations.extend(namess)
        break
    numbers = []
    for iteration in iterations:
        if "BestChangeInIteration" in iteration:
            numbers.append(int(iteration.split('_')[-1]))
    return int(max(numbers))

def getDocText(doctype, filePath):
    if doctype == "original":
        inputText = io.open(filePath + "/Orignal_Text", "r", errors="ignore").readlines()
        inputText = ''.join(str(e) + "" for e in inputText)
        return inputText
    elif doctype == "obfuscated":
        try:
            inputText = io.open(filePath + "/Obfuscated_Text", "r", errors="ignore").readlines()
            inputText = inputText[:-1]
            inputText = ''.join(str(e) + "" for e in inputText)
            return inputText
        except:
            inputText = io.open(filePath + "/BestChangeInIteration_" + str(24), "r", errors="ignore").readlines()
            inputText = inputText[:-1]
            inputText = ''.join(str(e) + "" for e in inputText)
            return inputText

def getMeteorScore(run):
    filePath = run.replace('Qualitative', '')
    reqRunName = filePath.split('/')[-1]
    filePath = '/'.join(filePath.split('/')[:-1]) + '/'
    allRuns = []
    for (_,_,files) in os.walk(filePath):
        allRuns.extend(files)
        break
    LogRunName = ''
    for i in allRuns:
        if reqRunName in i:
            LogRunName = i
            break

    data = pd.read_csv(filePath + "/" + str(LogRunName))
    return (data["meteorScore"].iloc[-1])


fileNames = getDirs(resultsFolder)
totalObfuscated = 0
meteorScores = []
count = 0
Obfuscated_Data = {}

if not os.path.exists("Obfuscated_Documents"):
    os.makedirs("Obfuscated_Documents")

for file in fileNames:
    print("Checking for file : ", file, "----(", count, "/", len(fileNames), ")")
    count+=1
    allRuns = getDirs(file)
    obfuscated_runs = []
    nonObfuscated_runs = []
    exceptionCounter = 0
    for run in allRuns:
        # orig_doc = getDocText('original', run)
        try:
            obfs_doc = getDocText('obfuscated', run)
        except Exception as e:
            exceptionCounter+=1
            print(e, exceptionCounter)
            continue


        try:
            if Obfuscated_or_Not(obfs_doc, (file.split('/')[-1]).split('_')[0]):
                print ('Obfuscated!')
            else:
                print ('Not Obfuscated!')
        except:
            if Obfuscated_or_Not(obfs_doc, ((file.split('/')[-1]).split('-')[1].split('.')[0])):
                obfuscated_runs.append((run, meteor_score))
            else:
                nonObfuscated_runs.append((run, meteor_score))

print("Total Non Obfuscated : ", totalObfuscated)
