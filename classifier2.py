# find . -type f -name foo\* -exec rm {} \;

import json
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import cPickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("data-sets/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)

# Read the test data
test = pd.read_csv("data-sets/testData.tsv", header=0, delimiter='\t', quoting=3)



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
    num_reviews = list_of_words.size
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
    with open('forest_dump/forest.pickle', 'wb') as f:
        cPickle.dump(forest, f)
    print "Forest dmuped!"


def pandas_dataframe_to_json(dataframe):
    d = [ 
        dict([
            (colname, row[i]) 
            for i,colname in enumerate(dataframe.columns)
        ])
        for row in dataframe.values
    ]
    return d

######################### Uncommment If cleaning training data need to happen again ################################

# clean_train_reviewsclean_train_words = clean_list_of_words(train["review"])
# dump_to_json("data.json")

######################### Uncommment If cleaning training data need to happen again ################################

# Loads clean data to be trained
clean_train_reviews = json.load(open('data.json','r'))


# Initialize the "CountVectorizer" object, which is scikit-learn's 
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word", tokenizer = None, preprocessor = None, max_features = 5000)

# fit_transform() does two functions: First , it transforms our training data 
# and learns the vocabulary; second; it transfoms our tranining data 
# into feature vectors. The input to fit_transform should be a list of strings

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array 
# train_data_features = train_data_features.toarray()


###################################### Uncommment If training need to happen again ##########################################

# print "Training the ramdom forest..."
# # Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators = 100) 

# # Fit the forest to the training set, using the bag of words as 
# # features and the sentiment labels as the response variable
# #
# # This may take a few minutes to run
# forest = forest.fit( train_data_features, train["sentiment"] )

# dump_pickle_file('forest_dump/forest.pickle', forest)

###################################### Uncommment If training need to happen again ##########################################

print "Reading forest...."
with open('forest_dump/forest.pickle', 'rb') as f:
    forest = cPickle.load(f)
print "Read complete!"

# Create an empty list and append the clean reviews one by one
print "Cleaning and parsing the test set movie reviews...\n"

clean_test_reviews = clean_list_of_words(test["review"])

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

d = pandas_dataframe_to_json(output)

with open('Bag_of_Words_model.json', 'w') as outfile:
    json.dump(d, outfile)


# Use pandas to write the comma-separated output file
# output.to_json( "Bag_of_Words_model.json")