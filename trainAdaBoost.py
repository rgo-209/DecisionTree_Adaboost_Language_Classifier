"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
import math
import pickle
import pandas
import random

col_names = ['hasZ',
             'avgWordLen',
             'dutchDiphtongs',
             'englishStopWords',
             'dutchStopWords',
             'englishCommonWords',
             'dutchCommonWords',
             'repeatedVowels',
             'repeatedConsonants',
             'ratioVowelsConsonants',
             'language',
             'weight']
totalNoOfEntriesValues = 0


class AdaModel:
    """"
        This class represents a trained AdaBoost model.
        The hypothesis vector and hypothesis weight vector
        are the 2 attributes of AdaBoost model.
        """
    def __init__(self,hypothesisVector, hypothesisWeightVector):
        self.hypothesisVector = hypothesisVector
        self.hypothesisWeightVector = hypothesisWeightVector


def weightOfStump(totalError):
    """"
        This functions is used to calculate the
        total say a stump will have based on the
        errors it committed.
        :param      totalError   the total error of a stump
        :return     weight a stump will have based on error
        """
    if totalError == 0:
        return 99999
    if totalError == 1:
        return -99999

    return (math.log((1 - totalError) / totalError)) / 2


def exponentValue(value):
    """"
        This functions is used to calculate the
        value of e raised to the value passed.
        :param      value       the value to raise with
        :return     result of e^value
        """
    try:
        return math.exp(value)
    except OverflowError:
        if value>=0:
            return float('inf')
        else:
            return float("-inf")


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


def getMostImportant(currentDataFrame):
    """"
        Function to return the most important attribute
        that can define te result.
        @:param  currentFeatures    the dataframe to find the attribute in
        @:return mostImportant      the most important feature in the data frame passed
        """

    sampleSpaceSize = len(currentDataFrame)
    maxGain = -999999
    mostImportant = ''

    noOfEnglish = len(currentDataFrame[currentDataFrame['language'] == 'en'])
    entropyEnglish = entropy(noOfEnglish / sampleSpaceSize)

    for currFeature in currentDataFrame.columns:
        if currFeature == 'language':
            continue
        if currFeature == 'weight':
            continue

        # Calculate all required values
        trueValuesEnglish = len(
            currentDataFrame[(currentDataFrame[currFeature] == True) & (currentDataFrame['language'] == 'en')])
        trueValuesDutch = len(
            currentDataFrame[(currentDataFrame[currFeature] == True) & (currentDataFrame['language'] == 'nl')])

        falseValuesEnglish = len(
            currentDataFrame[(currentDataFrame[currFeature] == False) & (currentDataFrame['language'] == 'en')])
        falseValuesDutch = len(
            currentDataFrame[(currentDataFrame[currFeature] == False) & (currentDataFrame['language'] == 'nl')])

        totalTrue = trueValuesEnglish + trueValuesDutch
        totalFalse = falseValuesEnglish + falseValuesDutch

        if totalTrue == 0 or totalFalse == 0:
            remainderEnglish = entropyEnglish
        else:
            # (pk+nk/p+n) B(pk/(pk+nk))
            probTrue = totalTrue / sampleSpaceSize
            probFalse = totalFalse / sampleSpaceSize

            probTrueEnglish = trueValuesEnglish / totalTrue
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


def learningAlgo(currentFeatures):
    """"
        This functions represents the learning
        algorithm used to find important attribute
        and assign weights and create a stump.
        :param      currentFeatures     the data to use for creating a stump
        :return     created stump data, mostImportantFeature of stump
                    and amt Of Say Of the Stump
        """

    # *************************** Select best feature **********************
    mostImportantFeature = getMostImportant(currentFeatures)

    print("Most Important feature selected: " + mostImportantFeature)

    trueWithDutch = currentFeatures[
        (currentFeatures[mostImportantFeature] == True) & (currentFeatures['language'] == 'nl')]

    falseWithEnglish = currentFeatures[
        (currentFeatures[mostImportantFeature] == False) & (currentFeatures['language'] == 'en')]

    trueWithDutchWeights = (trueWithDutch['weight']).sum()
    falseWithEnglishWeights = (falseWithEnglish['weight']).sum()
    totalIncorrectWeightSum = trueWithDutchWeights + falseWithEnglishWeights


    # *************************** Find Amount of say **********************

    amtOfSayOfCurrentStump = weightOfStump(totalIncorrectWeightSum)
    print("Sum of total incorrect weights: " + str(totalIncorrectWeightSum))
    print("\t\t\t\tAmount of say for current stump: " + str(amtOfSayOfCurrentStump))

    newWeightFactorForIncorrect = exponentValue(amtOfSayOfCurrentStump)
    newWeightFactorForCorrect = exponentValue(-amtOfSayOfCurrentStump)

    print("New Weight Factor For Incorrect values: " + str(newWeightFactorForIncorrect))
    print("New Weight Factor For Correct values: " + str(newWeightFactorForCorrect))

    tempFeatures = currentFeatures

    # *************************** Assign new weights **********************
    for index in range(len(tempFeatures)):
        if (tempFeatures.iloc[index, col_names.index(mostImportantFeature)] == True and
            tempFeatures.iloc[index, col_names.index('language')] == "en") or \
                (tempFeatures.iloc[index, col_names.index(mostImportantFeature)] == False and
                 tempFeatures.iloc[index, col_names.index('language')] == "nl"):
            tempFeatures.iloc[index, col_names.index('weight')] = tempFeatures.iloc[index, col_names.index('weight')] \
                                                                  + newWeightFactorForCorrect

        else:
            tempFeatures.iloc[index, col_names.index('weight')] = tempFeatures.iloc[
                                                                      index, col_names.index('weight')] \
                                                                  + newWeightFactorForIncorrect

    currentFeatures = tempFeatures

    totalWeightBeforeNormalization = (currentFeatures['weight']).sum()

    print("Total Weight Before Normalization: " + str(totalWeightBeforeNormalization))

    # *************************** Normalize the weights **********************

    for index in range(len(tempFeatures)):
        tempFeatures.iloc[index, col_names.index('weight')] = tempFeatures.iloc[index, col_names.index('weight')] \
                                                              / totalWeightBeforeNormalization

    currentFeatures = tempFeatures

    print("After normalization total weight: " + str((currentFeatures['weight']).sum()))

    return currentFeatures, mostImportantFeature, amtOfSayOfCurrentStump


def createNewFrameFromStump(decisionStump):
    """"
        This functions creates new data frame using
        a stump and gives more priority for classifying
        incorrect values predicted by current stump.
        :param      decisionStump     the decision stump to use
        :return     new data based on incorrect classification
                    of previous stump
        """
    newDataFrame = pandas.DataFrame(columns=col_names)

    # ********************* Adding incorrect predictions to new frame *************************
    for externalIndex in range(len(decisionStump)):
        randomNumberToCheck = random.random()
        cumulativeValue = 0
        for internalIndex in range(len(decisionStump)):
            cumulativeValue = cumulativeValue + decisionStump.iloc[internalIndex, col_names.index('weight')]
            if cumulativeValue >= randomNumberToCheck:
                newDataFrame.loc[externalIndex] = (decisionStump.iloc[internalIndex])
                break

    # ********************* Re-assigning weights in new frame *************************
    for index in range(len(newDataFrame)):
        newDataFrame.iloc[index, col_names.index('weight')] = 1/len(newDataFrame)

    return newDataFrame


def generateAdaBoost(currentDataFrame):
    """"
        This functions creates stumps one by ond
        comes up with a hypothesis vector and
        hypothesis weight vector.
        :param      currentDataFrame     the data to use
        :return     hypothesis vector and
                    hypothesis weight vector based on data
        """
    hypothesisVector = list()
    hypothesisWeightVector = list()

    # ************************ Create stumps one by one **********************
    for i in range(1, 6):
        print("Creating Stump: " + str(i))
        dataAfterStump, mostImportantFeature, amtOfSayOfCurrentStump = learningAlgo(currentDataFrame)
        hypothesisVector.append(mostImportantFeature)
        hypothesisWeightVector.append(amtOfSayOfCurrentStump)
        if i != 5:
            print("Updating data for next stump.")
            currentDataFrame = createNewFrameFromStump(dataAfterStump)
        print("\n\n")

    return hypothesisVector, hypothesisWeightVector


def trainAdaBoost(initialData, writeFile):
    """"
        Train the Ada Boost algorithm by calling functions
         and save it to a file specified.
        @:param  initialData    the data read from the file
        @:param  writeFile      the file to store results on
        @:return none
        """
    hypothesisVector, hypothesisWeightVector = generateAdaBoost(initialData)
    adaBoostModel = AdaModel(hypothesisVector, hypothesisWeightVector)

    print("Generated AdaBoost Model.")

    # Save AdaBoost model
    file = open(writeFile, 'wb')
    pickle.dump(adaBoostModel, file)
    print("\n\nAdaBoost model Saved.")

    return


def trainDataAdaBoost(allTrainData, resultFile):

    global totalNoOfEntriesValues
    totalNoOfEntriesValues = len(allTrainData)

    initialWeights = [1 / totalNoOfEntriesValues] * totalNoOfEntriesValues
    allTrainData['weight'] = initialWeights
    trainAdaBoost(allTrainData, resultFile)

