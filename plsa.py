import re
import numpy as np
from utils import normalize

"""
Author:
Alex Kong (https://github.com/hitalex)

Reference:
http://blog.tomtung.com/2011/10/plsa
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
        # ** ADD ** #
        # exclude words that appear in 90%+ of the documents
        # exclude words that are too (in)frequent
        discrete_set = set()
        for document in self.documents:
            if 'url' in document.words:
                print document
            for word in document.words:
                discrete_set.add(word)
                self.word_count += 1

        self.vocabulary = list(discrete_set)


    def read_priori(self, filename='priori.conf'):
        with open(filename, 'r') as fd:
            self.priori_weight = int(fd.readline().strip('\n'))
            for line in fd.readlines():
                line = line.strip('\n')
                tmp = line.split(',')
                topic_priori = np.zeros(len(self.vocabulary))
                for i in range(len(tmp)):
                    pair = tmp[i].strip()
                    word, count = pair.split(':')
                    word = word.strip()
                    count = int(count)
                    # BUG: cannot add non-exist word into vocabulary!!!
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
        # Get vocabulary and number of documents.
        # self.build_vocabulary()

        print 'url' in self.vocabulary
        # read priori for MAP estimation
        self.read_priori()
        print 'url' in self.vocabulary
        if self.priori_weight != 0:
            assert(len(self.priori) == number_of_topics)

        number_of_documents = len(self.documents)
        vocabulary_size = len(self.vocabulary)

        # build term-doc matrix
        term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
        # print self.vocabulary[-1]
        word_index_map = {v:k for k,v in enumerate(self.vocabulary)}
        for d_index, doc in enumerate(self.documents):
            # term_count = np.zeros(vocabulary_size, dtype = np.int)
            #print d_index
            for word in doc.words:
                #if word in self.vocabulary:
# BUG: tooooooo SLOW
                    #w_index = self.vocabulary.index(word)
                    #term_count[w_index] = term_count[w_index] + 1'
                w_index = word_index_map[word]
                term_doc_matrix[d_index][w_index] += 1
            #term_doc_matrix[d_index] = term_count

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
        """
        # for test, fixed values are assigned, where number_of_documents = 3, vocabulary_size = 15
        self.document_topic_prob = np.array(
        [[ 0.19893833,  0.09744287,  0.12717068,  0.23964181,  0.33680632],
         [ 0.27681925,  0.22971358,  0.1704416,   0.18248461,  0.14054095],
         [ 0.24768207,  0.25136754,  0.14392363,  0.14573845,  0.21128831]])

        self.topic_word_prob = np.array(
      [[ 0.02963563,  0.11659963,  0.06415405,  0.1291839 ,  0.09377842,
         0.09317023,  0.06140873,  0.023314  ,  0.09486251,  0.01538988,
         0.09189075,  0.06957687,  0.05015957,  0.05281074,  0.0140651 ],
       [ 0.09746902,  0.12212085,  0.07635703,  0.02799546,  0.0282282 ,
         0.03685356,  0.01256655,  0.03931912,  0.09545668,  0.00928434,
         0.11392475,  0.12089124,  0.02674909,  0.07219077,  0.12059333],
       [ 0.02209806,  0.05870101,  0.12101806,  0.03733935,  0.02550749,
         0.09906735,  0.0706651 ,  0.05619682,  0.10672434,  0.12259672,
         0.04218994,  0.10505831,  0.00315489,  0.03286002,  0.09682255],
       [ 0.0428768 ,  0.11598272,  0.08636138,  0.10917224,  0.05061344,
         0.09974595,  0.01647265,  0.06376147,  0.04468468,  0.01986342,
         0.10286377,  0.0117712 ,  0.08350884,  0.049046  ,  0.10327543],
       [ 0.02555784,  0.03718368,  0.10109439,  0.02481489,  0.0208068 ,
         0.03544246,  0.11515259,  0.06506528,  0.12720479,  0.07616499,
         0.11286584,  0.06550869,  0.0653802 ,  0.0157582 ,  0.11199935]])
        """
        # Run the EM algorithm
        for iteration in range(max_iter):
            print "Iteration #" + str(iteration + 1) + "..."
            print "E step:"
            #for d_index, document in enumerate(self.documents):
            #    print d_index, document
            #    for w_index in range(vocabulary_size):
            #        prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
            #        if sum(prob) == 0.0:
            #            print "d_index = " + str(d_index) + ",  w_index = " + str(w_index)
            #            print "self.document_topic_prob[d_index, :] = " + str(self.document_topic_prob[d_index, :])
            #            print "self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index])
            #            print "topic_prob[d_index][w_index] = " + str(prob)
            #            exit(0)
            #        else:
            #            normalize(prob)
            #        self.topic_prob[d_index][w_index] = prob

            A = np.zeros([number_of_documents, 1])
            B = np.zeros([1, vocabulary_size])
            for z in range(number_of_topics):
                A[:,0] = self.document_topic_prob[:,z]
                A_stacked = np.hstack([A]*vocabulary_size)
                B[0,:] = self.topic_word_prob[z,:]
                B_stacked = np.vstack([B]*number_of_documents)
                self.topic_prob[:,:,z] = A_stacked * B_stacked
                # print B
            sum_z = np.sum(self.topic_prob, axis=2)
            # print self.topic_prob[3,:,1]
            # print sum_z[np.where(sum_z <1e-6)]
            sum_z = np.dstack([sum_z]*number_of_topics)
            # print np.where(sum_z == 0)
            self.topic_prob[:,:,:] /= sum_z
                #print '%d sum' % (z)

            print "M step:"
            # update P(w | z)
            self.topic_word_prob = np.zeros(np.ma.shape(self.topic_word_prob))
            A = np.zeros([1,vocabulary_size])
            for d in range(number_of_documents):
                A[0,:] = term_doc_matrix[d, :]
                A_stacked = np.vstack([A]*number_of_topics)
                B = self.topic_prob[d, :, :]
                B = B.T
                #print np.shape(self.topic_word_prob), np.shape(A), np.shape(B)
                self.topic_word_prob[:,:] += (A_stacked * B)
                # print self.topic_word_prob[:,-1]
                # print term_doc_matrix[:,-1]

            sum_w = np.zeros([number_of_topics, 1])
            sum_w[:,0] = np.sum(self.topic_word_prob, axis=1)
            # print sum_w
            sum_w = np.hstack([sum_w]*vocabulary_size)
            if self.priori_weight != 0:
                sum_w += self.priori_weight
                self.topic_word_prob[:,:] += self.priori_weight*self.priori
            self.topic_word_prob[:,:] /= sum_w

            #for z in range(number_of_topics):
            #    for w_index in range(vocabulary_size):
            #        s = 0
            #        for d_index in range(len(self.documents)):
            #            count = term_doc_matrix[d_index][w_index]
            #            s = s + count * self.topic_prob[d_index, w_index, z]
            #        self.topic_word_prob[z][w_index] = s

            #    # MAP estimation, using a priori
            #    if self.priori_weight != 0:
            #        sum_w = sum(self.topic_word_prob[z])
            #        self.topic_word_prob[z] = self.topic_word_prob[z] + self.priori_weight*self.priori[z]
            #        for i in xrange(len(self.vocabulary)):
            #            self.topic_word_prob[z][i] = self.topic_word_prob[z][i]*1.0 / (sum_w + self.priori_weight)
            #    else:
            #        normalize(self.topic_word_prob[z])


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

            #for d_index in range(len(self.documents)):
            #    for z in range(number_of_topics):
            #        s = 0
            #        for w_index in range(vocabulary_size):
            #            count = term_doc_matrix[d_index][w_index]
            #            s = s + count * self.topic_prob[d_index, w_index, z]
            #        self.document_topic_prob[d_index][z] = s
#           #     print self.document_topic_prob[d_index]
#           #     assert(sum(self.document_topic_prob[d_index]) != 0)
            #    normalize(self.document_topic_prob[d_index])
