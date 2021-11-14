import pickle
import writeprintsStatic
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import operator
import io
import os
from pickle import load
import numpy as np
from keras.models import model_from_json
import pickle

def load_dataset(filename):
    return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

def load_model(filename):
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == 300:
                embeddings_index[word] = coefs
        except:
            # print(values)
            c = 1
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index



# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

def getData(datasetName, authorsRequired):
    
    picklesPath = 'Data/'
    with open(picklesPath+'X_train/X_train.pickle', 'rb') as handle:
        X_train = pickle.load(handle)

    with open(picklesPath+'X_test/X_test.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    
    
    return (X_train, X_test)
  
def getAllData(datasetName, authorsRequired):
    (X_train_all, X_test_all) = getData(datasetName, authorsRequired)
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for (filePath, filename, authorId, author, inputText) in X_train_all:
        X_train.append(inputText)
        y_train.append(authorId)
    for (filePath, filename, authorId, author, inputText) in X_test_all:
        X_test.append(inputText)
        y_test.append(authorId)

    return X_train, X_test, y_train, y_test
class Classifier:

    def __init__(self, classifierType, authorstoKeep, datasetName, indexNumber):

        self.classifierType = classifierType
        self.authorstoKeep = authorstoKeep
        self.datasetName = datasetName
        self.clf = None
        self.tokenizer = None
        self.MAX_SEQUENCE_LENGTH = None
        self.indexNumber = indexNumber

    def loadClassifier(self):

        if self.classifierType == 'ml':
            classifierName = "trainedModels/" + str(self.datasetName) + '-' + str(self.authorstoKeep) + '/trained_model_mutantxwp+dspan600.sav'
            # Original Model
            #classifierName = "/home/rzhai/Desktop/trained_model_obf.sav"
            #classifierName = "../../AuthorshipAttributionSystems/RFCWriteprintsStatic/trainedModels/" + str(self.datasetName) + '-' + str(self.authorstoKeep) + '/trained_model_dspan.sav'
            self.clf = pickle.load(open(classifierName, 'rb'))
        elif self.classifierType == 'dl':
            # load json and create model
            json_file = open("../../AuthorshipAttributionSystems/CNNWordEmbeddings/trainedModels/" + self.datasetName + "-" + str(self.authorstoKeep) + "/" + 'CNN_word_word_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.clf = model_from_json(loaded_model_json)
            # load weights into new model
            self.clf.load_weights("../../AuthorshipAttributionSystems/CNNWordEmbeddings/trainedModels/" + self.datasetName + "-" + str(self.authorstoKeep) + "/" + 'CNN_word_word_model.h5')
            print("Loaded model from disk")
            self.clf.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['acc'])
            trainLines, testLines, trainLabels, testLabels = getAllData(self.datasetName, self.authorstoKeep)
            
            # create tokenizer
            self.tokenizer = create_tokenizer(trainLines)
            self.MAX_SEQUENCE_LENGTH = max_length(trainLines)

    def getLabelAndProbabilities(self, document):

        if self.classifierType == 'ml':
            tempFile = open(str(self.indexNumber) + "tempText", "w")
            tempFile.write(document.documentText)
            tempFile.close()

            inputText = io.open(str(self.indexNumber) + "tempText" ,"r", errors="ignore").readlines()
            inputText = ''.join(str(e) + "" for e in inputText)


            # self.clf.predict([writeprintsStatic.calculateFeatures(document.documentText)])[0]
            document.documentAuthorProbabilites = {key: value for (key, value) in enumerate(
                list(self.clf.predict_proba([writeprintsStatic.calculateFeatures(inputText)])[0]))}
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree

            feature_names = [f'feature {i}' for i in range(555)]
            importances = self.clf.feature_importances_
            import pandas as pd
            forest_importances = pd.Series(importances, index=feature_names)
            forest_importances.to_excel("output.xlsx")
            plt.bar( range(len(importances)), importances)
            #forest_importances.nlargest(20).plot(kind='barh')
            '''             
            std = np.std([tree.feature_importances_ for tree in self.clf.estimators_],axis=0)
            indices = np.argsort(importances)
            
            plt.figure()
            plt.title("Feature importances")
            plt.barh(range(555), importances[indices],color="r", xerr=std[indices], align="center")
            plt.ylim([-1, 555])
            '''
            plt.savefig('feature_importance.png')
            
            '''
            fig = plt.figure(figsize=(15, 10))
            plot_tree(self.clf.estimators_[15], feature_names=range(555),class_names=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14'], 
                filled=True, impurity=True, 
                rounded=True)
            fig.savefig('figure.png', dpi=500)
            '''
            document.documentAuthor = self.clf.predict([writeprintsStatic.calculateFeatures(inputText)])[0]
            os.remove(str(self.indexNumber) + "tempText")

        elif self.classifierType == 'dl':
            sequences = self.tokenizer.texts_to_sequences([document.documentText])
            data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

            document.documentAuthorProbabilites = {key: value for (key, value) in enumerate(
                list(self.clf.predict_proba([np.array(data), np.array(data)])[0]))}

            document.documentAuthor, _ = max(document.documentAuthorProbabilites.items(), key=lambda x: x[1])
