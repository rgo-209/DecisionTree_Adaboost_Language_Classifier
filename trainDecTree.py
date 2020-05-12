"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
import math
from tree import Root
import pickle



def entropy(q):
    """"
        Function to return the entropy value of the probability passed.
        @:param     q   the probabilty of certain condition
        @:return    the entropy of the value passed
        """
    if q == 0 or q == 1:
        return 0
    else:
        return -1 * (q * math.log(q, 2) + (1 - q) * math.log((1 - q), 2))


def plurality_Value(currentFeatures):
    """"
        Function to return the plurality value of the dataframe.
        @:param     currentFeatures     the dataframe to check for
        @:return    the plurality value of the data frame passed
        """
    return currentFeatures['language'].mode()[0]


def classification(currentFeatures):
    """"
        Function to return the classification of the dataframe.
        @:param     currentFeatures     the dataframe to check for
        @:return    the classification of the data frame passed
        """
    return currentFeatures[currentFeatures['language'] == 'en']['language'].count()


def generateDecisionTree(treeLevel, currentFeatures, parentFeatures):
    """"
        Function to generate the decision tree with dataframe passed.
        @:param     treeLevel           the level of the current node
        @:param     currentFeatures     the current dataframe to use for making decision
        @:param     parentFeatures      the parent dataframe to use for making decision
        @:return    decTree             the decision tree formed
        """
    lenOfFrame = len(currentFeatures)
    classValue = classification(currentFeatures)

    if lenOfFrame==0:
        return Root(plurality_Value(parentFeatures))

    if classValue == lenOfFrame or classValue ==0 or len(currentFeatures.columns) ==1:
        return Root(plurality_Value(currentFeatures))

    else:
        mostImportantFeature = getMostImportant(currentFeatures)

        # Data when the most important feature is true
        trueFeatureData = currentFeatures[currentFeatures[mostImportantFeature] == True]
        # Data when the most important feature is false
        falseFeatureData = currentFeatures[currentFeatures[mostImportantFeature] == False]

        # Drop the most important feature
        trueFeatureData  = trueFeatureData.drop(labels=mostImportantFeature, axis=1)
        falseFeatureData = falseFeatureData.drop(labels=mostImportantFeature, axis=1)

        # Create root of tree
        decTree = Root(mostImportantFeature)

        # Get subtree when most important attribute is true
        tSubTree = generateDecisionTree(treeLevel + 1, trueFeatureData, currentFeatures)
        # Assign subtree
        decTree.true = tSubTree

        # Get subtree when most important attribute is false
        fSubTree = generateDecisionTree(treeLevel + 1, falseFeatureData, currentFeatures)
        # Assign subtree
        decTree.false = fSubTree

        # print(str(treeLevel) + "--" + mostImportantFeature + "-->" + str(tSubTree.attributeVal) + "--" + str(fSubTree.attributeVal))

        return decTree


def getMostImportant(currentFeatures):
    """"
        Function to return the most important attribute
        that can define te result.
        @:param  currentFeatures    the dataframe to find the attribute in
        @:return mostImportant      the most important feature in the data frame passed
        """

    sampleSpaceSize = len(currentFeatures)
    maxGain = -999999
    mostImportant = ''

    noOfEnglish = len(currentFeatures[currentFeatures['language'] == 'en'])
    entropyEnglish = entropy(noOfEnglish/sampleSpaceSize)

    for currFeature in currentFeatures.columns:
        if currFeature == 'language':
            continue

        # Calculate all required values
        trueValuesEnglish = len(currentFeatures[(currentFeatures[currFeature] == True)   & (currentFeatures['language'] == 'en')])
        trueValuesDutch = len(currentFeatures[(currentFeatures[currFeature] == True)   & (currentFeatures['language'] == 'nl')])

        falseValuesEnglish = len(currentFeatures[(currentFeatures[currFeature] == False) & (currentFeatures['language'] == 'en')])
        falseValuesDutch = len(currentFeatures[(currentFeatures[currFeature] == False) & (currentFeatures['language'] == 'nl')])

        totalTrue = trueValuesEnglish + trueValuesDutch
        totalFalse = falseValuesEnglish + falseValuesDutch

        if totalTrue == 0 or totalFalse == 0:
            remainderEnglish = entropyEnglish
        else:
                # (pk+nk/p+n) B(pk/(pk+nk))
            probTrue = totalTrue / sampleSpaceSize
            probFalse = totalFalse / sampleSpaceSize

            probTrueEnglish  = trueValuesEnglish / totalTrue
            probFalseEnglish = falseValuesEnglish / totalFalse

            # Calculate the remainder
            remainderEnglish = entropy(probTrueEnglish) * (probTrue) + entropy(probFalseEnglish) * (probFalse)

        currGain = entropyEnglish - remainderEnglish

        # Check whether the gain is maximum
        if currGain > maxGain:
            maxGain = currGain
            mostImportant = currFeature

    # Return the attribute which has maximum gain
    return mostImportant


def trainDecTree(initialData, writeFile):
    """"
            Train the decision tree by calling functions
             and save it to a file specified.
            @:param  initialData    the data read from the file
            @:param  writeFile      the file to store results on
            @:return none
            """
    # Generate decision tree
    decTree =  generateDecisionTree(0, initialData, initialData)
    print("Generated Decision tree.")

    # Save decision tree
    file = open(writeFile, 'wb')
    pickle.dump(decTree, file)
    print("Decision tree Saved.")

    return