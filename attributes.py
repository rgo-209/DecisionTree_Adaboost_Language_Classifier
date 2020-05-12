"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
import re

import pandas
from nltk.corpus import stopwords

vowels = ["a", "e", "i", "o", "u" ]

dutchDiphtongs = ['ae', 'ei', 'au', 'ai', 'eu', 'uw', 'eeuw', 'ieuw', 'ie', 'oe', 'ou', 'oei',
                   'ui', 'aai', 'oe', 'ooi', 'eeu', 'ieu' ]

dutchCommonWords = ['ik','je','het','de','is','dat','een','niet','en','wat','van','we','in','ze',
                    'op','te','hij','zijn','er','maar','me','die','heb','voor','met','als','ben',
                    'was','mijn','u','dit','aan','om','hier','naar','dan','jij','zo','weet','ja',
                    'kan','geen','nog','pas','wel','wil','moet','goed','hem','hebben','nee','heeft','waar',
                    'nu','hoe','ga','kom','uit','gaan','bent','haar','doen','ook','mij','over','of','daar',
                    'zou','al','jullie','zal','bij','ons','gaat','hebt','meer','waarom','iets','deze','laat',
                    'had','doe','m','moeten','wie','jou','alles','denk','kunnen','eens','echt','weg','door','man',
                    'toch','zien','oké','alleen','nou','dus','nooit']

englishCommonWords = ['a','about','all','also','and','as','at','be','because',
                      'but','by','can','come','could','day','do','even','find',
                      'first','for','from','get','give','go','have','he','her',
                      'here','him','his','how','I','if','in','into','it','its',
                      'just','know','like','look','make','man','many','me','more',
                      'my','new','no','not','now','of','on','one','only','or','other',
                      'our','out','people','say','see','she','so','some','take',
                      'tell','than','that','the','their','them','then','there',
                      'these','they','thing','think','this','those','time','to',
                      'two','up','use','very','want','way','we','well','what',
                      'when','which','who','will','with','would','year','you','your']

stopWordsEnglish = (set(stopwords.words('english')))
stopWordsDutch = (set(stopwords.words('dutch')))


def hasDutchDiphtongs(sentence):
    """"
        Check if the sentence has Dutch Diphtongs
        @:param  sentence    the sentence to check for
        @:return true if sentence has Dutch Diphtongs
                else false
        """
    sentence = sentence.split(" ")
    for word in sentence:
        for diphtongs in dutchDiphtongs:
            if diphtongs in word:
                return True
    return False


def hasZ(sentence):
    """"
        Check if the sentence has Z in it
        @:param  sentence    the sentence to check for
        @:return true if sentence has z else false
        """
    for word in sentence:
        if 'z' in word:
            return True
    return False


def hasDutchCommonWord(sentence):
    """"
        Check if the sentence has English Common Words
        @:param  sentence    the sentence to check for
        @:return true if sentence has English Common Words
                else false
        """
    sentence = sentence.split(" ")
    for word in sentence:
        if word in dutchCommonWords:
                return True
    return False


def hasEnglishCommonWord(sentence):
    """"
        Check if the sentence has English Common Words
        @:param  sentence    the sentence to check for
        @:return true if sentence has English Common Words
                else false
        """
    sentence = sentence.split(" ")
    for word in sentence:
        if word in englishCommonWords:
            return True
    return False


def hasDutchStopWords(sentence):
    """"
        Check if the sentence has Dutch stop Words
        @:param  sentence    the sentence to check for
        @:return true if sentence has Dutch stop Words
                else false
        """
    sentence = sentence.split(" ")
    for word in sentence:
        if word in stopWordsDutch:
            return True

    return False


def hasEnglishStopWords(sentence):
    """"
        Check if the sentence has English stop Words
        @:param  sentence    the sentence to check for
        @:return true if sentence has English stop Words
                else false
        """
    sentence = sentence.split(" ")
    for word in sentence:
        if word in stopWordsEnglish:
            return True

    return False


def ratioVowelsConsonants(sentence):
    """"
        Count number of vowels and consonants
        @:param  sentence           the sentence to check for
        @:return vowelsCount        the number of vowels
        @:return consonantsCount    the number of consonants
        """
    sentence = sentence.split(" ")
    vowelsCount = 0
    consonantsCount = 0
    for word in sentence:
        for ch in word:
            if ch.isalpha():
                if (ch in vowels):
                    vowelsCount= vowelsCount + 1
                else:
                    consonantsCount=consonantsCount + 1
    if consonantsCount != 0 and (vowelsCount/consonantsCount)>0.725:
        return True
    else:
        return False


def wordLength(sentence):
    """"
        Check what is average word length of sentence
        @:param  sentence   the sentence to check for
        @:return true       if the average word length > 5.5
                else return false
        """
    sentence = sentence.split(' ')

    totalWords = len(sentence)
    totalLength = 0

    for word in sentence:
        totalLength+=len(word)

    if totalLength/totalWords > 7 :
        return True
    else:
        return False


def hasRepeatedVowels(sentence):
    """"
        Check if sentence has repeated vowels
        @:param  sentence   the sentence to check for
        @:return true       if the sentence has repeated
                            vowels else return false
        """
    sentence = sentence.split(" ")
    cnt=0
    for word in sentence:
        if ('aa') in word or \
                ('ee') in word or \
                ('ii') in word or \
                ('oo') in word or\
                ('uu') in word :
            cnt = 1 + cnt
            if cnt > 2:
                return True
    return False


def hasRepeatedConsonants(sentence):
    """"
        Check if sentence has repeated consonants
        @:param  sentence   the sentence to check for
        @:return true       if the sentence has repeated
                            consonants else return false
        """
    cnt = 0
    sentence = sentence.split(" ")
    for word in sentence:
        for i in range(len(word)-2):
            if word[i].isalpha():
                    if word[i] not in vowels:
                        if word[i] == word[i + 1]:
                            cnt= cnt+1
    if cnt>3:
        return True

    return False


def removePunctuationToLowercae(sentence):
    """"
        Removes punctuation from sentence
        @:param  sentence   the sentence to check for
        @:return sentence   sentence after removing punctuation
                            and changing to lower case
        """

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    for mark in sentence.lower():
        if mark in punctuations:
            sentence = sentence.replace(mark, "")
            sentence = re.sub('[0-9]+', '', sentence)
    return sentence.lower()


def getAttributesToTrain(initDataframe):

    col_names =  ['hasZ',
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

    allSentences = initDataframe['sentence']
    allLanguages = initDataframe['language']

    finalData = pandas.DataFrame(columns=col_names)


    for i in range (len(allLanguages)):
        sentence = allSentences[i]
        attributes = {
            'hasZ': hasZ(sentence),
            'avgWordLen': wordLength(sentence),
            'dutchDiphtongs': hasDutchDiphtongs(sentence),
            'englishStopWords': hasEnglishStopWords(sentence),
            'dutchStopWords' : hasDutchStopWords(sentence),
            'englishCommonWords': hasEnglishCommonWord(sentence),
            'dutchCommonWords': hasDutchCommonWord(sentence),
            'repeatedVowels': hasRepeatedVowels(sentence),
            'repeatedConsonants': hasRepeatedConsonants(sentence),
            'ratioVowelsConsonants': ratioVowelsConsonants(sentence),
            'language': allLanguages[i]
        }
        finalData = finalData.append(attributes, ignore_index=True)

    return finalData


def getAttributesToTest(initDataframe):

    col = ['hasZ',
                 'avgWordLen',
                 'dutchDiphtongs',
                 'englishStopWords',
                 'dutchStopWords',
                 'englishCommonWords',
                 'dutchCommonWords',
                 'repeatedVowels',
                 'repeatedConsonants',
                 'ratioVowelsConsonants']

    allSentences = initDataframe['sentence']

    finalData = pandas.DataFrame(columns=col)

    for i in range(len(allSentences)):
        sentence = allSentences[i]
        attributes = {
            'hasZ': hasZ(sentence),
            'avgWordLen': wordLength(sentence),
            'dutchDiphtongs': hasDutchDiphtongs(sentence),
            'englishStopWords': hasEnglishStopWords(sentence),
            'dutchStopWords': hasDutchStopWords(sentence),
            'englishCommonWords': hasEnglishCommonWord(sentence),
            'dutchCommonWords': hasDutchCommonWord(sentence),
            'repeatedVowels': hasRepeatedVowels(sentence),
            'repeatedConsonants': hasRepeatedConsonants(sentence),
            'ratioVowelsConsonants': ratioVowelsConsonants(sentence)
        }
        finalData = finalData.append(attributes, ignore_index=True)

    return finalData