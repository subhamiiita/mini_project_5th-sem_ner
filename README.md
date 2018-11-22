# About




This is an implementation using (linear chain) conditional random fields (CRF) in python 3.7 for named entity recognition (NER) in hindi. It uses the python-crfsuite library as its basis.It can handle the labels PER, LOC, ORG and O(others). Scores are expected to be a bit lower for other labels than PER, because training data has more person in it. The implementation achieved an F1 score for PER was 0.923 and an F1 score for LOC was 0.838 and for ORG ,it was 0.500.


# Used features

The CRF implementation uses following features:

- Word Features:current word,previous word and next word
- POS tag of current word,previous word and next word
- Suffixes of current word

# Requirements

## Libraries/Data

- python 3 or above (only tested on that version)
- python-crfsuite
- nltk 
- pos tagged hindi data

# Installtion guide:

- sudo apt-get update

- sudo apt-get install python3-pip

- sudo apt install python3-sklearn

- pip3 install sklearn-crfsuite

- pip3 install nltk

# Files attached:

## Dataset:


- train_ner.txt:

	Contains 200 words(sample)

- test_accuracy.txt

	Contains 50 words(sample)

- test_file.txt

	Contains 1 hindi sentece given by user

# Actual code:

### mini_project_ner.py
(to identify ner of hindi sentences)



	input_file:
	
	train_ner.txt(contains words,it's pos and it's ner)
	test_file.txt(user gives hindi words which contains word and its POS)

	output_file:
	
	predicted ner in file output_file.txt. 

### mini_project_ner_accuracy.py
(to calculate precison,recall,f1 score and support)
	
	
	input_file:
	
	train_ner.txt(contains words,it's pos and it's ner)
	test_accuracy.txt(contains words,it's pos and it's ner)

	output_file:
	
	predicted ner in file output_file_predicted.txt.


# Contact details(group members):

- subham raj(iit2016010),7992256326

- ankit kumar(iit2016024),8229880027

- deepanshu goyal(iit2016037),7571848905
