This is a small example of using a classifier to determine the author of a document whose authorship is in question. 

## Background

Paul Collins wrote an [essay in *The New Yorker* in 2013](https://www.newyorker.com/books/page-turner/poes-debut-hidden-in-plain-sight) about using stylometric analysis to help determine the author of "A Fragment," an 1827 short story published in *North American* magazine. The story was published under the name of William Henry Poe, but it sounded a lot like it was written by his brother, Edgar Allan Poe. 

Collins uses JGAAP to find the likely author. Here, we'll use a much simpler approach, using word frequency with Scikit-learn's classifier.

## In this repository

- docs/ 
	- Contains writing samples from Edgar Allan Poe, William Henry Poe, James Fenimore Cooper, Nathaniel Hawthorne, Washington Irving, and John Neal (all mentioned by Collins in his essay as the set of 2 likely authors + 4 distractor authors; but not the actual test texts that Collins used in his attempt)
- test/
	- Contains the short story in question, "A Fragment"
- top1000.txt
	- The top 1000 words in the English language (from a separate corpus), minus *I* and *a* since they are one-letter words ignored by the vectorizer 
- train-test-classifier.py
	- The goods: the Python script that trains a classifier on docs/ and then tests it out on test/anon_a-fragment.txt, outputting the suspected author

## To run this script

### Install required libraries

You'll need Scikit-learn, joblib, and Numpy. Need these? From the command line, run:

```pip install sklearn```

```pip install joblib```

```pip install numpy```

### Run the script

From command line:

```python3 train-test-classifier.py```

In IDLE:

Open train-test-classifier.py in IDLE and run the module 

### Inputs / outputs

Inputs: 17 .txt files to train, 1 .txt file to test (all needed inputs are included i this repo)

On-screen outputs: classifier accuracy, then predicted author of "A Fragment." It should look like this:

> ```Classifier accuracy on training document set:  1.000```

> ```Predicted author of anonymous document:  epoe```

File output: traindocs_tf-array.csv (in case you want to take a look behind the scenes. Put the header and row names in yourself: the header is the contents of top1000.txt, and the row names are the contents of docs/ in alpha order)