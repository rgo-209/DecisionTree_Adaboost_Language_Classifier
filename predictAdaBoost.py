"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
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
             'language']


def predictAdaBoost(adaBoostModel, allTestData):
    """"
           Function to test the result of a line based on current frame passed.
           @:param  adaBoostModel       the adaBoostModel to test in
           @:param  allTestData         the data file to test for
           @:return the result prediction
           """

    # print("Predicting using AdaBoost model.")

    for externalIndex in range(len(allTestData)):
        currentLineAttributes = allTestData.iloc[externalIndex]
        englishCount = 0
        dutchCount = 0

        for internalIndex in range(len(adaBoostModel.hypothesisVector)):
            currentFeature = adaBoostModel.hypothesisVector[internalIndex]
            if currentLineAttributes[col_names.index(currentFeature)]:
                if adaBoostModel.hypothesisWeightVector[internalIndex] >= 0:
                    # print("add to english")
                    englishCount = englishCount + adaBoostModel.hypothesisWeightVector[internalIndex]
                else:
                    # print("add to dutch")
                    dutchCount = dutchCount + abs(adaBoostModel.hypothesisWeightVector[internalIndex])
            else:
                if adaBoostModel.hypothesisWeightVector[internalIndex] >= 0:
                    # print("add to english")
                    dutchCount = dutchCount + adaBoostModel.hypothesisWeightVector[internalIndex]
                else:
                    # print("add to dutch")
                    englishCount = englishCount + abs(adaBoostModel.hypothesisWeightVector[internalIndex])

        if englishCount >= dutchCount:
            # print(str(externalIndex) + "  en")
            print("en")
        else:
            # print(str(externalIndex) + "  nl")
            print("nl")
