import nltk
from nltk.corpus import stopwords
import numpy as np
from bs4 import BeautifulSoup
from timeit import default_timer as timer

STOP_WORDS = stopwords.words("english")

def preprocess_dataset(file_path_dataset):
    print "Started the preprocessing of %s" % file_path_dataset

    start = timer()
    reviews, bag_of_words = extractallsentences(file_path_dataset)
    end = timer()

    print "Finished preprocessing in %s" % (end - start)

    bag_of_words = sorted(list(bag_of_words))
    word_dict = dict([(word, word_id) for word_id, word in enumerate(bag_of_words)])

    print "Started creating the document-word matrix"
    start = timer()
    doc_words, doc_sentence_words = create_doc_word_matrix(reviews, word_dict)
    doc_words = doc_words.astype(int)
    end = timer()

    print "Finished creating the document-word matrix in %s" % (end - start)

    n_sent = 0
    for d in doc_sentence_words:
        n_sent += len(d)
    print 'number of sentences: ', n_sent
    return reviews, bag_of_words, doc_words, doc_sentence_words


def extractallsentences(file_path_dataset):
    parser = BeautifulSoup(open(file_path_dataset, 'r'), 'lxml')

    number_of_reviews = 0
    bag_of_words = set()
    reviews = []

    start = timer() # Keep track of processing time

    for review in parser.find_all("review_text"):
        if number_of_reviews % 1000 == 0:
            print '{0} has been parsed'.format(number_of_reviews)
        cleaned_sentences = []
        reviewsentences = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(review.text)]

        for sentence in reviewsentences:
            cleaned_sentence = preprocess_sentence(sentence)
            [bag_of_words.add(word) for word in cleaned_sentence]
            cleaned_sentences.append(cleaned_sentence)

        reviews.append(cleaned_sentences)
        number_of_reviews += 1

    end = timer()

    print "Processed reviews from XML using 'tree.findall' in %s time."% (end - start)
    print "Number of reviews processed: %d" % number_of_reviews
    print "Number of words in bag %d" % len(bag_of_words)

    return reviews, bag_of_words

def preprocess_sentence(sentence):
    cleaned_sentence = []

    for word in sentence:
        word = word.strip().lower()
        if word == None or word in STOP_WORDS or word=='' or len(word) <= 2:
            continue
        else:
            cleaned_sentence.append(word)

    return cleaned_sentence

def create_doc_word_matrix(documents, bag_of_words):
    doc_word_mat = np.zeros((len(documents), len(bag_of_words)))
    doc_sentence_words = []

    for doc_id, doc_content in enumerate(documents):
        doc_sent = []
        for sentence in doc_content:
            sent_words = []
            for word in sentence:
                word_id = bag_of_words[word]
                doc_word_mat[doc_id, word_id] += 1
                sent_words.append(word_id)
            doc_sent.append(sent_words)
        doc_sentence_words.append(doc_sent)

    return doc_word_mat, doc_sentence_words

