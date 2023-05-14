from keras_preprocessing import text
import numpy as np
import re
import nltk
import spacy

from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


class FeaturesComputation:


    def __init__(self):
        # word filters
        self.features = []
        self.labels = []
        self.filters = ",.?!\"'`;:-()&$"
        self.upperChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.specialChars = "~@#$%^&*-_=+><[]{}/\|"
        self.letters = "abcdefghijklmnopqrstuvwxyz"
        self.digit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.bigrams = ['th','he','in','er','an','re','nd','at','on','nt','ha','es','st' ,'en','ed','to','it','ou','ea','hi','is','or','ti','as','te','et' ,'ng','of','al','de','se','le','sa','si','ar','ve','ra','ld','ur']
        self.trigram = ["the", "and", "ing", "her", "hat", "his", "tha", "ere", "for", "ent", "ion", "ter", "was", "you", "ith",
                    "ver", "all", "wit", "thi", "tio"]
        self.tags = ['ADJ', 'ADP', 'ADV','AUX', 'CONJ','CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
        

    def totalWords(self, data):
        # Calculating total words in input.
        return len(text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" "))

    def avgWordLen(self, data):
        # Calculate average word length
        words = text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" ")
        word_len = []
        for w in words:
            word_len.append(len(w))
        return np.mean(word_len)

    def totalShortWords(self, data):
        # Calculate total number of short words
        words = text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" ")
        short_words = []
        for w in words:
            if len(w) < 4:
                short_words.append(w)
        return len(short_words)

    def characterCount(self, data):
        # Calculate total characters
        input = data.lower()
        return len(str(input))

    def digitsByPer(self, data):
        # Calculate the total percentage of digits out of total characters
        input = data.lower()
        total_char = len(str(input))
        total_digits = list([1 for i in str(input) if i.isnumeric()==1]).count(1)

        return total_digits/total_char

    def upperCaseCharbyPer(self, data):
        # Calculate the percentage of uppercase characters out of total characters
        data = data.replace(" ", "")
        Uchars = self.upperChars 
        total_char = len(str(data))
        u_chars = [1 for i in str(data) if i in Uchars].count(1)

        return u_chars/total_char

    def countSpecialChar(self, data):
        # Calculate the counts of sprecial character and return frequency of special character
        s_char_dict = {}
        s_char = self.specialChars

        for i in range(0, len(s_char)):
            s_char_dict[s_char[i]] = 0
            for j in str(data):
                if s_char[i] == j:
                    s_char_dict[s_char[i]] = s_char_dict[s_char[i]] + 1

        # Vectorization of counts
        count_vector = [0] * len(s_char)
        total = sum(list(s_char_dict.values())) + 1

        for k in range(0, len(s_char)):
            count_vector[k] = s_char_dict[s_char[k]]/total

        return np.array(count_vector)

    def letterCount(self, data):
        # Calculate the counts of letters and return frequency of letters
        letters = self.letters
        data = str(data).lower()
        data = data.lower().replace(" ", "")
        letter_dict = {}

        for i in range(0, len(letters)):
            letter_dict[letters[i]] = 0
            for j in str(data):
                if letters[i] == j:
                    letter_dict[letters[i]] = letter_dict[letters[i]] + 1

        # vectorization of counts
        count_vector = [0] * len(letters)
        total = sum(list(letter_dict.values())) + 1

        for k in range(0, len(letters)):
            count_vector[k] = letter_dict[letters[k]]/total

        return np.array(count_vector)

    def digitsCount(self, data):
        # Calculate the counts of digits and return frequency of digits
        digit = self.digit
        digitsCount = {}

        for d in digit:
            digitsCount[str(d)] = 0

        digitsintext = re.findall('\d', str(data))

        for d in digit:
            digitsCount[str(d)] = digitsintext.count(str(d))

        # dig = np.array(digitsCount.values())
        total = self.characterCount(data)
        dig = [value/total for key, value in digitsCount.items()]
        # return np.divide(dig, characterCount(data))
        return dig
        # return np.divide(dig, characterCount(data))

    def mcLetterBigram(self, data):
        # Calculate the counts of bigrams and return frequency of bigrams
        bigr = self.bigrams
        bg_count = {}

        for b in bigr:
            bg_count[b] = 0

        input = text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" ")

        for word in input:
            for i in range(0, len(word)-1):
                bg = (word[i:i+2]).lower()
                if bg in bigr:
                    bg_count[bg] = bg_count[bg] + 1

        total = sum(list(bg_count.values()))

        return [float(bg_count[b]/total) for b in bigr]

    def mcLettersTrigram(self, data):
        # Calculate the counts of trigrams and return frequency of trigrams
        tri = self.trigram
        tri_count = {}

        for t in tri:
            tri_count[t] = 0

        input = text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" ")

        for word in input:
            for i in range(0, len(word)-2):
                tr = (word[i:i+3]).lower()
                if tr in tri:
                    tri_count[tr] = tri_count[tr] + 1

        total = sum(list(tri_count.values()))

        return [float(tri_count[t]/total) for t in tri]

    def legomena(self, data):
        input = text.text_to_word_sequence(data, filters=self.filters, lower=True, split=" ")
        word_freq = nltk.FreqDist(word for word in input)
        h = [key for key, value in word_freq.items() if value == 1]
        d = [key for key, value in word_freq.items() if value == 2]
        try:
            return list((len(h)/len(input.split()), len(d)/len(input.split())))
        except:
            return [0, 0]

    def functionWordsFreq(self, data):
        fun_path = str(PROJECT_ROOT) + '/util/functionWord.txt'
        # fun_path = 'functionWord.txt'
        fun_words = open(fun_path, "r").readlines()
        fun_words = [i.strip("\n") for i in fun_words]
        words = text.text_to_word_sequence(data, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True, split=" ")
        freq = [words.count(i) for i in fun_words]
        return freq

    def postagFreq(self, data):
        nlp = spacy.load('en_core_web_sm')
        d = nlp(str(data))
        pos_tag = []

        for t in d:
            pos_tag.append(str(t.pos_))

        tags = self.tags 
        t = [tag for tag in pos_tag]
        return list(tuple(t.count(tag)/len(t) for tag in tags))

    def PuncCharFrequency(self, data):

        inputText = str(data).lower()
        in_p = inputText.lower().replace(" ", "")

        p_path = str(PROJECT_ROOT) + '/util/punctuation.txt'
        p_words = open(p_path, "r").readlines()
        p_words = [i.strip("\n") for i in p_words]

        count = []
        for i in range(0, len(p_words)):
            count.append(in_p.count(p_words[i]))

        total_count = sum(count)+1

        return [count[i]/total_count for i in range(0, len(count))]



    def calFeatures(self, extension, label):
        self.features.extend([i for i in extension])
        self.labels.extend([label for i in extension])

    def getFeatures(self, input):
        
        # Feature set 1
        self.calFeatures([self.totalWords(input)], 'totalWords')
        self.calFeatures([self.avgWordLen(input)], 'averageWordLength')
        self.calFeatures([self.totalShortWords(input)], 'totalShortWords')
        
        # Feature set 2
        self.calFeatures([self.characterCount(input)], 'characterCount')
        self.calFeatures([self.digitsByPer(input)], 'digitsByPer')
        self.calFeatures([self.upperCaseCharbyPer(input)], 'upperCaseCharbyPer')
        
        # Feature set 3
        self.calFeatures(self.countSpecialChar(input), 'countSpecialCharacter')
        
        # Feature set 4
        self.calFeatures(self.letterCount(input), 'LetterCount')
        
        # Feature set 5
        self.calFeatures(self.digitsCount(input), 'DigitsCount')
        
        # Feature set 6 
        self.calFeatures(self.mcLetterBigram(input), 'mostCommonLetterBigram')
        
        # Feature set 7
        self.calFeatures(self.mcLettersTrigram(input), 'mostcommonLettersTrigram')
        
        # Feature set 8
        self.calFeatures(self.legomena(input), 'legomena')
        
        # Feature set 9
        self.calFeatures(self.functionWordsFreq(input), 'functionWordsFreq')
        
        # Feature set 10
        self.calFeatures(self.postagFreq(input), 'PosTag Frequency')
        
        # Feature set 11
        self.calFeatures(self.PuncCharFrequency(input), 'PunctuationCharactersFrequency')

        return self.features