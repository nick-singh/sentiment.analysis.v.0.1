import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist



#reading pre-labeled input and splitting into lines
posSentences_input = open('rt-polaritydata/rt-polarity.pos', 'r')
negSentences_input = open('rt-polaritydata/rt-polarity.neg', 'r')


def evaluate_features(features_select):
	posSentences = re.split(r'\n', posSentences_input.read())
	negSentences = re.split(r'\n', negSentences_input.read())

	posFeatures = []
	negFeatures = []

	for i in posSentences:
		posWords = re.findall(r"[\w']+|[.,!?;]", i)
		posWords = [features_select(posWords), 'pos']
		posFeatures.append(posWords)
	for i in negSentences:
		negWords = re.findall(r"[\w']+|[.,!?;]", i)
		negWords = [features_select(negWords), 'neg']
		negFeatures.append(negWords)

	#selects 3/4 of the features to be used for training and 1/4 to be used for testing
	posCutoff = int(math.floor(len(posFeatures)*3/4))
	negCutoff = int(math.floor(len(negFeatures)*3/4))
	trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
	testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

	#trains a Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(trainFeatures)	

	#initiates referenceSets and testSets
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)	

	#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
	for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		predicted = classifier.classify(features)
		testSets[predicted].add(i)	

	#prints metrics to show how well the feature selection did
	print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
	print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
	print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
	print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
	print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
	classifier.show_most_informative_features(10)


#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)