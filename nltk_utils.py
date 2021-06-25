import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    #return nltk.word_tokenize(sentence)
    tokenlist=[]
    doc=nlp(sentence)
    for token in doc:
        tokenlist.append(token.text)
    return tokenlist



def lemm(word):
    """
    Lemmatization = find the root form of the word
   
    """
    #return stemmer.stem(word.lower())
    doc=nlp(word)
    for token in doc:
        return token.lemma_.lower()





def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [lemm(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
