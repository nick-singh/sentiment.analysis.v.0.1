from TwitterSearch import *
import json
try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_keywords(['Taylor Swift']) # let's define all words we would like to have a look for
    tso.set_language('en') # we want to see German tweets only
    tso.set_include_entities(False) # and don't give us all those entity information

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        access_token = "717397609-eVR9kuq30tAhbY26NNpruYDgkimVdB185ciVYgFt",
		access_token_secret = "Sx2vxEyh0GQgkQiBtTOyHockz5aEX6kPcKjDosrCFMX9O",
		consumer_key = "96AQF2axAeEipYRYVKLk7pV88",
		consumer_secret = "sR4boPFTcDxV8IRjjvUuZavMvoeZ2Hd2ob7ykNN3YTheaZz65H"
     )
    tweets_data = []
     # this is where the fun actually starts :)
    for i,(tweet) in enumerate(ts.search_tweets_iterable(tso)):
        if(i<10):
        	tweets_data.append(tweet)
        else:
        	break

    with open('data-sets/temp.json', 'w') as outfile:
        json.dump(tweets_data,outfile)

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)

