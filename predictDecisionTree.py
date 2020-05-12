"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
import pandas

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

def testOutput(root, features):
    """"
           Function to test the result of a line based on current frame passed.
           @:param  root        the root of decision tree
           @:param  features    the current features to test
           @:return the result of traversing the tree
           """
    if root.attributeVal == 'en':
        return "en"
    if root.attributeVal == 'nl':
        return "nl"
    else:
        attr = root.attributeVal
        val = col_names.index(attr)
        # print(val)

        if features[val]== True:
            # If the feature is true traverse true subtree
            return testOutput(root.true,features)
        else:
            # If the feature is true traverse true subtree
            return testOutput(root.false,features)


def predictDecTree(decTree, allTestData):
    it = allTestData.iterrows()

    cnt= 0

    # Predict the result line by line
    for i in it:
        i = i[1]
        currentFeatures = []
        for index in range(len(i)):
            currentFeatures.append((i.get(col_names[index])))

        currentFeatures = pandas.array(currentFeatures)
        print(""+testOutput(decTree, currentFeatures))
        cnt = cnt+1
