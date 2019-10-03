#This script trains a classifier on documents from six authors,
#then uses the classifier on a short story ("A Fragment") whose
#authorship is in question. By @robincamille for a demo.

#Inputs: 17 .txt files to train, 1 .txt file to test

#On-screen outputs: classifier accuracy, then predicted author of "A Fragment"

#File output: traindocs_tf-array.csv (term frequency array for training docs
# - see line 68)

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy

#Set up document lists
#The docs whose authorship we know
traindocs = []
traindocsfiles = ['cooper_deerslayer.txt',\
    'cooper_last-of-the-mohicans_chap-1-2.txt',\
    'cooper_pioneers.txt',\
    'epoe_cask-of-amontillado_eleonora_domain.txt',\
    'epoe_gold-bug_tell-tale-heart.txt',\
    'epoe_mask-of-the-red-death_bernice.txt',\
    'hawthorne_birthmark.txt',\
    'hawthorne_scarlet-letter_part-1.txt',\
    'hawthorne_scarlet-letter_part-2.txt',\
    'irving_sketchbook-of-geoffrey-crayon.txt',\
    'irving_sleepy-hollow.txt',\
    'irving_the-alhambra.txt',\
    'neal_children.txt',\
    'neal_goody-gracious.txt',\
    'neal_pickings-and-stealings.txt',\
    'wpoe_recollections.txt',\
    'wpoe_the-pirate.txt']

for doc in traindocsfiles:
    with open('docs/' + doc, 'r') as fulltext:
        fulltext = fulltext.read()
        traindocs.append(fulltext)

#Docment labels, aka who wrote them, in order
targets = ['cooper','cooper','cooper',\
    'epoe','epoe','epoe',\
    'hawthorne','hawthorne','hawthorne',\
    'irving','irving','irving',\
    'neal','neal','neal',\
    'wpoe','wpoe']

#Creates word-count array for a given text
#Use only vocab of top 1k most frequent words (given)
with open('top1000.txt','r') as vocdoc:
    voc = [w[:-1] for w in vocdoc.readlines()]

def wordfreq(docs):
    '''wordcount(documentList) -> converts collection of documents to term frequency array'''
    tf = TfidfVectorizer(vocabulary=voc,use_idf=False)
    alltexts = []
    for doc in docs:
        alltexts.append(doc)
    tfarray = tf.fit_transform(alltexts)
    return tfarray

#Set up term frequency arrays for training document set
traintf = wordfreq(traindocs).toarray()

#Output this array as a CSV, in case you want to take a look behind the scenes
#Header is contents of top1000.txt, column order is traindocsfiles
numpy.savetxt("traindocs_tf-array.csv",traintf,delimiter="\t")

#Set up classifier
gnb = GaussianNB()
preds = gnb.fit(traintf, targets).predict(traintf)
scoretrain = "%.3f" % gnb.score(traintf, targets)
print("Classifier accuracy on training document set: ", scoretrain)

#Save the classifier
classif = joblib.dump(gnb,'myclassifier') 

#The doc whose authorship we don't know, or do know and want to
#use to test the classifier
testdocs = []
testdocsfiles = ['anon_a-fragment.txt']
for doc in testdocsfiles:
    with open('test/' + doc, 'r') as fulltext:
        fulltext = fulltext.read()
        testdocs.append(fulltext)

#Set up term freq for anonymous doc(s)
anontf = wordfreq(testdocs).toarray()

#Use trained classifier on new text(s), return prediction
gnbtest = joblib.load(classif[0]) #reuse saved classifier
predicttest = gnb.predict(anontf)
print("Predicted author of anonymous document: ", predicttest[0])
