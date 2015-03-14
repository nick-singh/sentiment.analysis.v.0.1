import classifier2
import json
import cPickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


print "reading reviews..."
train = pd.read_csv("data-sets/labeledTrainData.tsv", header=0,delimiter="\t", quoting=3)

# Read the test data
test = pd.read_csv("data-sets/testData.tsv", header=0, delimiter='\t', quoting=3)
print "Completed..."

print "Loading tweets"
tweets_data_path = 'data-sets/twitter_stream.json'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
print "tweets loaded"

tweets = pd.DataFrame()

tweets['id'] = map(lambda tweet: tweet['id'], tweets_data)
tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

# Initialize the "CountVectorizer" object, which is scikit-learn's 
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word", tokenizer = None, preprocessor = None, max_features = 5000)


# Loads clean data to be trained
clean_train_reviews = json.load(open('data.json','r'))
 
# fit_transform() does two functions: First , it transforms our training data 
# and learns the vocabulary; second; it transfoms our tranining data 
# into feature vectors. The input to fit_transform should be a list of strings

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array 
train_data_features = train_data_features.toarray()


forest = classifier2.read_pickle_file('forest_dump/forest.pickle')

# Create an empty list and append the clean reviews one by one
clean_test_reviews = classifier2.clean_list_of_words(tweets["text"])

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":tweets["id"], "sentiment":result} )

d = classifier2.pandas_dataframe_to_json(output)
# json_data = json.dumps(d)

sentiments = {
    "pos" : 0,
    "neg" : 0
}


for data in d:
    # print data['sentiment']
    if data['sentiment'] == 1:
        sentiments['pos']+=1
    else:
        sentiments['neg']+=1

print sentiments
# with open('Bag_of_Words_model.json', 'w') as outfile:
#     json.dump(d, outfile)


# Use pandas to write the comma-separated output file
# output.to_json( "Bag_of_Words_model.json")