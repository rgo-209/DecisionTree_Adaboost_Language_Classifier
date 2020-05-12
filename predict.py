"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
import sys
import pandas
import pickle

from trainAdaBoost import AdaModel
from attributes import getAttributesToTest
from predictAdaBoost import predictAdaBoost
from predictDecisionTree import predictDecTree
from tree import Root


def predict(trainedFile, testingFile):
    """"
           Function to predict the result of lines from a file.
           @:param  trainedFile        the trained file to access
           @:param  testing file       the file to test
           @:return none
           """
    col = ['sentence']
    testData = pandas.read_csv(testingFile, names=col, sep='\n')
    testFeature = testData[col]

    # convert data into attributes
    allTestData = getAttributesToTest(testFeature)

    # read the trained model
    try:
        file = open(trainedFile, 'rb')
        loadedData = pickle.load(file)
    except IOError:
        print("File not found..!!\nExiting..!!")
        sys.exit(-1)

    # If the file used has decision tree model
    if type(loadedData) is Root:
        predictDecTree(loadedData, allTestData)
    # If the file used has AdaBoost model
    elif type(loadedData) is AdaModel:
        predictAdaBoost(loadedData, allTestData)
    else:
        # Else exit
        print("Wrong type of file chose..!!")
        sys.exit(-1)


def main():
    # print("Using trained model from file: ",sys.argv[1])
    # print("Reading testing data from: ", sys.argv[2])
    predict(sys.argv[1], sys.argv[2])



if __name__ == '__main__':
    try:
        main()
    except IOError:
        print("Error occurred..!!")