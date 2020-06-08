"""
PROBLEM STATEMENT:-

    To build a model to accurately classify a piece of news as REAL or FAKE. Using sklearn,  build a TfidfVectorizer
    on the provided dataset. Then, initialize a PassiveAggressive Classifier and fit the model. In the end,
    the accuracy score and the confusion matrix tell us how well our model fares. On completion, create a GitHub
    account and create a repository. Commit your python code inside the newly created repository.

Author: Soumitra Edake
"""

# imports
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset into the dataframe df
# read_csv() is used to ready csv files
# All the labels, read from datasets, are stored in variable 'labels'
df = pd.read_csv("news.csv")
labels = df.label

# Splitting the dataframe df in 80-20 percent.
# 80% will be passed to training function.
# rest 20# will be passed to testing function.
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# TfidfVectorizer() convert a collection of raw documents to a matrix of TF-IDF features
# 'english' is passed to _check_stop_list which returns appropriate required stop list
# max_df = 0.7 means "ignore terms that appear in more than 70% of the documents"
vectors = TfidfVectorizer(stop_words='english', max_df=0.7)

# fit_transform() learns the vocabulary and idf and returns term-document matrix
# transform() transform documents to document-term matrix.
t_train = vectors.fit_transform(x_train)
t_test = vectors.transform(x_test)

# max_iter is used to set the maximum number of passes over the training data (aka epochs)
# fit() fits the linear model with Passive Aggressive algorithm
pred_model = PassiveAggressiveClassifier(max_iter=50)
pred_model.fit(t_train, y_train)

# predict() predicts the class labels for samples in t_test
# accuracy_score() is used to measure model's accuracy by passing it known values and predicted values
y_pred = pred_model.predict(t_test)
score = accuracy_score(y_test, y_pred)

# confusion_matrix() generates confusion matrix to evaluate the accuracy of a classification
# confusion matrix is a table with two rows and two columns that reports the number of false positives,
# false negatives, true positives, and true negatives.
con_mat = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Finally, printing out the result
print("Accuracy : " + str(score))
print("Confusion Matrix:-\n" + str(con_mat))
print("True Positives : " + str(con_mat[0][0]) + "\nTrue Negatives : " + str(con_mat[1][1]))
print("False Positives : " + str(con_mat[1][0]) + "\nFalse Negatives : " + str(con_mat[0][1]))


"""
Sample Output :-

Accuracy : 0.9329123914759274
Confusion Matrix:-
[[592  46]
 [ 39 590]]
True Positives : 592
True Negatives : 590
False Positives : 39
False Negatives : 46

"""