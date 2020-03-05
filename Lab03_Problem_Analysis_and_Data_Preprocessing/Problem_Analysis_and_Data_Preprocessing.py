
# coding: utf-8

# # Data Preprocessing
# 
# The goal of this lab is to introduce you to data preprocessing techniques in order to make your data suitable for applying a learning algorithm.
# 
# ## 1. Handling Missing Values
# 
# A common (and very unfortunate) data property is the ocurrence of missing and erroneous values in multiple features in our dataset.
# Download the dataset and corresponding information from the <a href="http://www.cs.uni-potsdam.de/ml/teaching/ss15/ida/uebung02/abalone.csv">course website</a>.
# 
# To determine the age of a abalone snail you have to kill the snail and count the annual
# rings. You are told to estimate the age of a snail on the basis of the following attributes:
# 1. type: male (0), female (1) and infant (2)
# 2. length in mm
# 3. width in mm
# 4. height in mm
# 5. total weight in grams
# 6. weight of the meat in grams
# 7. drained weight in grams
# 8. weight of the shell in grams
# 9. number of annual rings (number of rings +1, 5 yields age)
# 
# However, these data is incomplete. Missing values are marked with −1.

# In[2]:

import pandas as pd

# load data 
df = pd.read_csv("http://www.cs.uni-potsdam.de/ml/teaching/ss15/ida/uebung02/abalone.csv")
df.columns=['type','length','width','height','total_weight','meat_weight','drained_weight','shell_weight','num_rings']
df.head()


# ### Exercise 1.1
# 
# Compute the mean of all positive numbers of each numeric column and the counts of each category.

# In[3]:

##################
#INSERT CODE HERE#
##################


# ### Exercise 1.2
# 
# Compute the median of all positive numbers of each numeric column.

# In[13]:

##################
#INSERT CODE HERE#
##################


# ### Exercise 1.3
# 
# Handle the missing values in a way that you find suitable. Argue your choices.

# In[14]:

##################
#INSERT CODE HERE#
##################


# ### Exercise 1.4
# 
# Perform Z-score normalization on every column (except the type of course!)

# In[15]:

##################
#INSERT CODE HERE#
##################


# ## 2. Preprocessing text (Optional)
# 
# One possible way to transform text documents into vectors of numeric attributes is to use the TF-IDF representation. We will experiment with this representation using the 20 Newsgroup data set. The data set contains postings on 20 different topics. The classification problem is to decide which of the topics a posting falls into. Here, we will only consider postings about medicine and space.

# In[5]:

from sklearn.datasets import fetch_20newsgroups

categories = ['sci.med', 'sci.space']
raw_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
print 'The index of each category is: {}'.format([(i,target) for i,target in enumerate(raw_data.target_names)])


# Check out some of the postings, might find some funny ones!

# In[6]:

idx = np.random.randint(0, len(raw_data.data))
print 'This is a {} email.\n'.format(raw_data.target_names[raw_data.target[idx]])
print 'There are {} emails.\n'.format(len(raw_data.data))
print(raw_data.data[idx])


# Lets pick the first 10 postings from each category

# In[7]:

idxs_med = np.flatnonzero(raw_data.target == 0)
idxs_space = np.flatnonzero(raw_data.target == 1)
idxs = np.concatenate([idxs_med[:10],idxs_space[:10]])
data = np.array(raw_data.data)
data = data[idxs]


# <a href="http://www.nltk.org/">NLTK</a> is a toolkit for natural language processing. Take some time to install it and go through this <a href="http://www.slideshare.net/japerk/nltk-in-20-minutes">short tutorial/presentation</a>.
# 
# The downloaded package below is a tokenizer that divides a text into a list of sentences, by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.

# In[8]:

import nltk
import itertools
nltk.download('punkt')

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in data]
vocabulary_size = 1000
unknown_token = 'unknown'


# In[9]:

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())


# In[10]:

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])


# ### Exercise 2.1
# 
# Code your own TF-IDF representation function and use it on this dataset. (Don't use code from libraries. Build your own function with Numpy/Pandas)

# In[12]:

from sklearn.feature_extraction.text import CountVectorizer
countvec = CountVectorizer()
df = pd.DataFrame(countvec.fit_transform(data).toarray(), columns=countvec.get_feature_names())

def tfidf(df):
    
    ##################
    #INSERT CODE HERE#
    ##################
    
    return rep

rep = tfidf(df)

# Check if your implementation is correct
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(norm=None, smooth_idf=False, use_idf=True)
X_train = pd.DataFrame(vectorizer.fit_transform(data).toarray(), columns=countvec.get_feature_names())
answer=['No','Yes']
if rep is not None:
    print 'Is this implementation correct?\nAnswer: {}'.format(answer[1*np.all(X_train == rep)])


# In[ ]:



