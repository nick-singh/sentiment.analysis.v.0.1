# find . -type f -name foo\* -exec rm {} \;

import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import cPickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer



def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  



# Function accepts list of words and clean each row
def clean_list_of_words(list_of_words):
    
    # Get the number of reviews based on the datagrame column size
    num_reviews = len(list_of_words)
    #
    # Initialize an empty list to hold the clean reviews
    clean_train_words = []
    #
    print "Cleaning and parsing the training set items...\n"
    #
    clean_train_words = []
    for i in xrange( 0, num_reviews ):
        # If the index is evenly divisible by 1000, print a message
        if( (i+1)%1000 == 0 ):
            print "item %d of %d\n" % ( i+1, num_reviews )                                                                    
        clean_train_words.append( review_to_words( list_of_words[i] ))
    return clean_train_words


def dump_to_json(file_name_and_path, data):    
    with open(file_name_and_path, 'w') as outfile:
        json.dump(data, outfile)


def dump_pickle_file(file_name_and_path, forest):
    print "Forest trained, dumping...."
    with open(file_name_and_path, 'wb') as f:
        cPickle.dump(forest, f)
    print "Forest dmuped!"

def read_pickle_file(file_name_and_path):
    print "Reading forest...."
    with open(file_name_and_path, 'rb') as f:
        forest = cPickle.load(f)    
    print "Read complete!"
    return forest


def pandas_dataframe_to_json(dataframe):
    d = [ 
        dict([
            (colname, row[i]) 
            for i,colname in enumerate(dataframe.columns)
        ])
        for row in dataframe.values
    ]
    return d

