import re
import numpy as np
from utils import normalize

"""
Author:
Alex Kong (https://github.com/hitalex)

Reference:
http://blog.tomtung.com/2011/10/plsa

Author:
    dragonly<liyilongko@163.com> (https://github.com/dragonly)
Contribution:
    Optimized EM algorithm(actually rewritten).
    Instead of using several nested loops in python code, I reimplemented them using
    matrix operations using numpy library functions, which is implemented in C code.
"""

np.set_printoptions(threshold='nan')

class Document(object):

    '''
    Splits a text file into an ordered list of words.
    '''

    # List of punctuation characters to scrub. Omits, the single apostrophe,
    # which is handled separately so as to retain contractions.
    PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']

    # Carriage return strings, on *nix and windows.
    CARRIAGE_RETURNS = ['\n', '\r\n']

    # Final sanity-check regex to run on words before they get
    # pushed onto the core words list.
    WORD_REGEX = "^[a-z']+$"


    def __init__(self, filepath):
        '''
        Set source file location, build contractions list, and initialize empty
        lists for lines and words.
        '''
        self.filepath = filepath
        #self.file = open(self.filepath)
        self.lines = []
        self.words = []


    def split(self, STOP_WORDS_SET):
        '''
        Split file into an ordered list of words. Scrub out punctuation;
        lowercase everything; preserve contractions; disallow strings that
        include non-letters.
        '''
        self.file = open(self.filepath)
        try:
            self.lines = [line for line in self.file]
        finally:
            self.file.close()

        for line in self.lines:
            words = line.split(' ')
            for word in words:
                clean_word = self._clean_word(word)
                if clean_word and (clean_word not in STOP_WORDS_SET) and (len(clean_word) > 1): # omit stop words
                    self.words.append(clean_word)


    def _clean_word(self, word):
        '''
        Parses a space-delimited string from the text and determines whether or
        not it is a valid word. Scrubs punctuation, retains contraction
        apostrophes. If cleaned word passes final regex, returns the word;
        otherwise, returns None.
        '''
        word = word.lower()
        for punc in Document.PUNCTUATION + Document.CARRIAGE_RETURNS:
            word = word.replace(punc, '').strip("'")
        return word if re.match(Document.WORD_REGEX, word) else None


class Corpus(object):

    '''
    A collection of documents.
    '''

    def __init__(self):
        '''
        Initialize empty document list.
        '''
        self.documents = []
        self.word_count = 0

        self.priori = []
        # pseudo document size
        self.priori_weight = 0

    def add_document(self, document):
        '''
        Add a document to the corpus.
        '''
        self.documents.append(document)


    def build_vocabulary(self):
        '''
        Construct a list of unique words in the corpus.
        '''
        discrete_set = set()
        for document in self.documents:
            for word in document.words:
                discrete_set.add(word)
                self.word_count += 1

        self.vocabulary = list(discrete_set)


    def read_priori(self, filename='priori.conf'):
        with open(filename, 'r') as fd:
            self.priori_weight = int(fd.readline().strip('\n'))
            for line in fd.readlines():
                line = line.strip('\n')
                tmp = line.split(',')zxcg
                topic_priori = np.zeros(len(self.vocabulary))
                for pair in tmp:
                    word, count = pair.split(':')
                    word = word.strip()
                    count = int(count)
                    # WARNNING: cannot add non-exist word into vocabulary!!!
                    if word in self.vocabulary:
                        index = self.vocabulary.index(word)
                        topic_priori[index] = count
                normalize(topic_priori)
                self.priori.append(topic_priori)
            self.priori = np.array(self.priori)


    def plsa(self, number_of_topics, max_iter):

        '''
        Model topics.
        '''
        print "EM iteration begins..."

        self.read_priori()
        if self.priori_weight != 0:
            assert(len(self.priori) == number_of_topics)

        number_of_documents = len(self.documents)
        vocabulary_size = len(self.vocabulary)

        # build term-doc matrix
        term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
        word_index_map = {v:k for k,v in enumerate(self.vocabulary)}
        for d_index, doc in enumerate(self.documents):
            for word in doc.words:
                #if word in self.vocabulary:
# WARNNING: search for word in self.vocabulary is tooooooo SLOW, maybe because it's O(n) operation
                w_index = word_index_map[word]
                term_doc_matrix[d_index][w_index] += 1

        # Create the counter arrays.
        self.document_topic_prob = np.zeros([number_of_documents, number_of_topics], dtype=np.float) # P(z | d)
        self.topic_word_prob = np.zeros([number_of_topics, vocabulary_size], dtype=np.float) # P(w | z)
        self.topic_prob = np.zeros([number_of_documents, vocabulary_size, number_of_topics], dtype=np.float) # P(z | d, w)

        # Initialize
        print "Initializing..."
        # randomly assign values
        self.document_topic_prob = np.random.random(size = (number_of_documents, number_of_topics))
        for d_index in range(len(self.documents)):
            normalize(self.document_topic_prob[d_index]) # normalize for each document
        self.topic_word_prob = np.random.random(size = (number_of_topics, vocabulary_size))
        for z in range(number_of_topics):
            normalize(self.topic_word_prob[z]) # normalize for each topic
        # Run the EM algorithm
        for iteration in range(max_iter):
            print "Iteration #" + str(iteration + 1) + "..."
            print "E step:"

            A = np.zeros([number_of_documents, 1])
            B = np.zeros([1, vocabulary_size])
            for z in range(number_of_topics):
                A[:,0] = self.document_topic_prob[:,z]
                A_stacked = np.hstack([A]*vocabulary_size)
                B[0,:] = self.topic_word_prob[z,:]
                B_stacked = np.vstack([B]*number_of_documents)
                self.topic_prob[:,:,z] = A_stacked * B_stacked
            sum_z = np.sum(self.topic_prob, axis=2)
            sum_z = np.dstack([sum_z]*number_of_topics)
            self.topic_prob[:,:,:] /= sum_z

            print "M step:"
            # update P(w | z)
            self.topic_word_prob = np.zeros(np.ma.shape(self.topic_word_prob))
            A = np.zeros([1,vocabulary_size])
            for d in range(number_of_documents):
                A[0,:] = term_doc_matrix[d, :]
                A_stacked = np.vstack([A]*number_of_topics)
                B = self.topic_prob[d, :, :]
                B = B.T
                self.topic_word_prob[:,:] += (A_stacked * B)

            sum_w = np.zeros([number_of_topics, 1])
            sum_w[:,0] = np.sum(self.topic_word_prob, axis=1)
            # print sum_w
            sum_w = np.hstack([sum_w]*vocabulary_size)
            if self.priori_weight != 0:
                sum_w += self.priori_weight
                self.topic_word_prob[:,:] += self.priori_weight*self.priori
            self.topic_word_prob[:,:] /= sum_w

            # update P(z | d)

            self.document_topic_prob = np.zeros(np.ma.shape(self.document_topic_prob))
            A = np.zeros([number_of_documents,1])
            for w in range(vocabulary_size):
                A[:,0] = term_doc_matrix[:, w]
                A_stacked = np.hstack([A]*number_of_topics)
                B = self.topic_prob[:,w,:]
                self.document_topic_prob[:,:] += A_stacked * B

            sum_zd = np.zeros([number_of_documents, 1])
            sum_zd[:,0] = np.sum(self.document_topic_prob, axis=1)
            sum_zd = np.hstack([sum_zd]*number_of_topics)
            self.document_topic_prob[:,:] /= sum_zd

# TODO: add computation for posteriori after each iteration
