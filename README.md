# mini_project_5th-sem_ner


installtion guide:
sudo apt-get update

sudo apt-get install python3-pip

sudo apt install python3-sklearn

pip3 install sklearn-crfsuite

pip3 install nltk

files attached:

Dataset:
train_ner.txt:

Contains 200 words(sample)

test_accuracy.txt

Contains 50 words(sample)

test_file.txt

Contains 1 hindi sentece given by user

ACTUAL CODE:
mini_project_ner.py
(to identify ner of hindi sentences)



	input_file:
	train_ner.txt(contains words,it's pos and it's ner)
	test_file.txt(user gives hindi words which contains word and its POS)

	output_file:
	predicted ner in file output_file.txt.

mini_project_ner_accuracy.py
(to calculate precison,recall,f1 score and support)
	
	
	input_file:
	train_ner.txt(contains words,it's pos and it's ner)
	test_accuracy.txt(contains words,it's pos and it's ner)

	output_file:
	predicted ner in file output_file_predicted.txt.


contact details(group members):

subham raj(iit2016010),7992256326

ankit kumar(iit2016024),8229880027

deepanshu goyal(iit2016037),7571848905
