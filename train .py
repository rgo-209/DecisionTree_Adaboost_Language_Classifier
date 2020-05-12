"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""

import pandas
import sys
import attributes
from trainAdaBoost import trainDataAdaBoost
from trainDecTree import trainDecTree


def defineAttributes(trainFile):
    """"
        Get the attributes for the given file.
        @:param  trainFile      the file to read and train data on
        @:return dataframe of attributes values
        """
    col_names = ['language', 'sentence']
    try:
        readData = pandas.read_csv(trainFile, sep='|', names=col_names)
    except IOError:
        print("File not found..!!\nExiting..!!")
        sys.exit(-1)

    print("Read the data from file.")

    return attributes.getAttributesToTrain(readData)



def trainUsingDecTree(trainFile, resultFile):
    """"
        Train the decision tree by calling functions
         and save it to a file specified.
        @:param  trainFile      the file to read and train data on
        @:param  resultFile     the file to store results on
        @:return none
        """

    print("Train decision tree from "+trainFile+" and save to "+resultFile)

    allTrainData = defineAttributes(trainFile)

    print("Extracted all attributes from the data.")

    trainDecTree(allTrainData, resultFile)




def trainUsingAdaBoost(trainFile, resultFile):
    print("Train adaboost from "+trainFile+" and save to "+resultFile)

    allTrainData = defineAttributes(trainFile)

    print("Extracted all attributes from the data.")

    trainDataAdaBoost(allTrainData, resultFile)



def main():

    print("Reading training data from: ", sys.argv[1])
    print("Saving trained model to: ",sys.argv[2])

    if sys.argv[3]=='dt':
        print("\n********** Using Decision Tree **********")
        trainUsingDecTree(sys.argv[1], sys.argv[2])

    elif str(sys.argv[3])=="ada":
        print("\n********** Using Adaboost **********")
        trainUsingAdaBoost(sys.argv[1], sys.argv[2])

    else:
        print("Wrong Command line parameters..!!")


if __name__ == '__main__':
    main()