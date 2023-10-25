'''
BMI 550
Assigment2_ PART1_ Naive Bayes Model
@author Seyedeh Somayyeh Mousavi
Department of Biomedical Informatics
School of Medicine
Emory University
email: bmemousavi@gmail.com

Answer these question:
-Implement an automatic classifier
-Cross-validate on the training set
-Tune hyperparameters
-Apply some sort of ensemble classification
-Compare at least 5 classifiers + a Naive Bayes baseline -Engineer at least 4 features + n-grams
-Identify the best classifier & feature set combination

Date: 08/20/2023
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from nltk.probability import FreqDist
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB
import sys
import nltk
import codecs
import string

stop_words = set (stopwords.words('english'))
nltk.download('punkt') 

st = stopwords.words('english')
stemmer = PorterStemmer()

def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df

def preprocess_text(text):
    if isinstance(text, str):
        # If the input is a string, perform text cleaning
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        return text
    else:
        # Handle cases where the input is not a string (e.g., NaN)
        return ''

word_clusters = {}
def loadwordclusters():
    infile = open('./50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    if isinstance(sent, str):  # Check if 'sent' is a string
        sent = sent.lower()
        terms = nltk.word_tokenize(sent)
        cluster_string = ''
        for t in terms:
            if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
        return str.strip(cluster_string)
    else:
        return ''  

def grid_search_hyperparam_space(params, classifier, x_train, y_train):
        custom_scorer = make_scorer(f1_score, average='micro')
        grid_search = GridSearchCV(estimator=classifier, param_grid=params, refit=True, cv=3, return_train_score=False, scoring = custom_scorer)
        grid_search.fit(x_train, y_train)
        return grid_search
    
def get_tokens(text):
    # Tokenize the text using NLTK's word_tokenize or any other method
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t in stop_words]
    return tokens if tokens else []  # Return an empty list if tokens are None    
    
def print_frequency_dist(df):
    # Create a dictionary to store tokens for each category
    tokens = defaultdict(list)

    # Iterate through the DataFrame and tokenize the text
    for index, row in df.iterrows():
        doc_label = row['target']
        doc_text = row['text']
        doc_text = preprocess_text(doc_text)
        doc_tokens = get_tokens(doc_text)
        tokens[doc_label].extend(doc_tokens)

    # Calculate and print word frequency distributions for each category
    for category_label, category_tokens in tokens.items():
        print(f"Category {category_label} Word Frequencies:")
        fd = FreqDist(category_tokens)
        print(fd.most_common(30))  # Print the 30 most common words for each category   

def preprocess_num_falls_column(column):
    """
    Preprocess a column containing 'num_falls_6_mo' or similar data.

    Args:
        column (pandas.Series): The column to be preprocessed.

    Returns:
        pandas.Series: The preprocessed column.
    """
    # Replace "3 or more" with 3
    column.replace('3 or more', 3, inplace=True)

    # Handle "None" values: Replace with 0
    column = column.apply(lambda x: 0 if x == "None" else x)

    # Convert the column to numeric
    column = pd.to_numeric(column)
    return column

def calculate_classification_metrics(true_labels, predicted_labels):
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Micro-averaged F1 score
    micro_f1 = f1_score(true_labels, predicted_labels, average='micro')
    
    # Macro-averaged F1 score
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    # Precision and Recall for each class
    precision = precision_score(true_labels, predicted_labels, average=None)
    recall = recall_score(true_labels, predicted_labels, average=None)
    
    # Classification report with precision, recall, F1-score, and support for each class
    report = classification_report(true_labels, predicted_labels)
    print('Accuracy:', accuracy)
    print('Micro-F1:', micro_f1)
    print('Macro-F1:', macro_f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Classification Report:\n', report)

    
    return {
        'Accuracy': accuracy,
        'Micro-F1': micro_f1,
        'Macro-F1': macro_f1,
        'Precision': precision,
        'Recall': recall,
        'Classification Report': report
    } 
# Feature extraction
def sentence_to_vector(sentence):
    words = [word for word in sentence if word in word_vectors]
    if not words:
        return np.zeros(1000)  # If no words in vocabulary, return zeros
    return np.mean([word_vectors[word] for word in words], axis=0)

def convert_gender_to_binary(gender):
    if gender == 'Female':
        return 0
    elif gender == 'Male':
        return 1
    else:
        return gender
      
##########################################################
if __name__ == '__main__':
    
    #Load the train data
    f_path = './fallreports_2023-9-21_train.csv'
    df_train= loadDataAsDataFrame(f_path)
    df1 = pd.DataFrame({'text': df_train['fall_description'], 'target': df_train ['fog_q_class']})
    # print_frequency_dist(df1)
    
    words_to_add = ["patient", "landed", "fell", "balance", "got", "back", 'walking']
    for word in words_to_add:
        stop_words.add(word)
    # print_frequency_dist(df1)
  
    training_texts = df_train['fall_description']
    training_gender = ((df_train['gender'].apply(convert_gender_to_binary)).values).reshape(-1, 1)
    training_classes = df_train ['fog_q_class']
    training_age = df_train['age_at_enrollment'].values
    training_number_falls = preprocess_num_falls_column (df_train['num_falls_6_mo'])
  
    word_clusters = loadwordclusters()
    
    #PREPROCESS THE DATA
    training_texts_preprocessed = []
    training_clusters = []
    training_texts_tokenized = []

    for tr in training_texts:
        training_texts_preprocessed.append(preprocess_text(tr))
        training_clusters.append(getclusterfeatures(preprocess_text(tr)))
        training_texts_tokenized.append(preprocess_text(tr))
        
    # Features
    model = Word2Vec(training_texts_tokenized, vector_size=1000, window=5, min_count=1, sg=0)
    # Extract word vectors
    word_vectors = model.wv
    training_age = training_age.reshape(-1, 1)
    training_number_falls = (training_number_falls.values).reshape(-1, 1)
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
    clustervectorizer = CountVectorizer(ngram_range=(1,3), max_features=1000)
    tfidf_vectorizer = TfidfVectorizer()
    
    training_word2vec = [sentence_to_vector(sentence) for sentence in training_texts_tokenized] #list
    training_data_vector = vectorizer.fit_transform(training_texts_preprocessed).toarray()
    training_cluster_vectors = clustervectorizer.fit_transform(training_clusters).toarray()
    training_tfidf_vectorizer = tfidf_vectorizer.fit_transform(training_texts_preprocessed).toarray()
   
    training_data_vectors = np.concatenate((training_data_vector, training_cluster_vectors, training_word2vec, 
                                            training_tfidf_vectorizer, training_age, training_number_falls, training_gender), axis=1)
    
    print("Shape of training_data_vectors:", training_data_vectors.shape)
    print("Shape of training_classes:", np.array(training_classes).shape)
    ###########################################################################
    #Load the test data
    f_path = './fallreports_2023-9-21_test.csv'
    df_test= loadDataAsDataFrame(f_path)
    test_texts = df_test['fall_description']
    test_gender = ((df_test['gender'].apply(convert_gender_to_binary)).values).reshape(-1, 1)
    test_classes = df_test['fog_q_class']
    test_age = df_test['age_at_enrollment'].values
    test_age = test_age.reshape(-1, 1)
    test_number_falls = preprocess_num_falls_column(df_test['num_falls_6_mo'])
    test_number_falls = (test_number_falls.values).reshape(-1, 1)


    # PREPROCESS THE TEST DATA
    test_texts_preprocessed = []
    test_clusters = []
    test_texts_tokenized = []

    for tr in test_texts:
        test_texts_preprocessed.append(preprocess_text(tr))
        test_clusters.append(getclusterfeatures(preprocess_text(tr)))
        test_texts_tokenized.append(preprocess_text(tr))

    test_word2vec = [sentence_to_vector(sentence) for sentence in test_texts_tokenized] #list
    test_data_vector = vectorizer.transform(test_texts_preprocessed).toarray()
    test_cluster_vectors = clustervectorizer.transform(test_clusters).toarray()
    test_tfidf_vectorizer = tfidf_vectorizer.transform(test_texts_preprocessed).toarray()

    # Concatenate the test data vectors
    test_data_vectors = np.concatenate((test_data_vector, test_cluster_vectors, test_word2vec,
                                        test_tfidf_vectorizer, test_age, test_number_falls, test_gender), axis=1)
    print("Shape of testing_data_vectors:", test_data_vectors.shape)
    print("Shape of testing_classes:", np.array(test_classes).shape)
    
    # 4. Train a Naive Bayes Model (Multinomial Naive Bayes in this example)
    
    from sklearn.preprocessing import MinMaxScaler
    NV = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
    NV.fit(training_data_vectors, training_classes)

    # 5. Model Evaluation
    y_pred = NV.predict(test_data_vectors)
    accuracy = accuracy_score(test_classes, y_pred)
    conf_matrix = confusion_matrix(test_classes, y_pred)
    calculate_classification_metrics(test_classes ,y_pred)  
    report = classification_report(test_classes, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)
    '''
   Cross validation Results 
   
   Naive Bayes Model 
   Accuracy: 0.704225352112676
   Micro-F1: 0.704225352112676
   Macro-F1: 0.6839092643629425
   Precision: [0.64150943 0.88888889]
   Recall: [0.94444444 0.45714286]
   Classification Report:
                  precision    recall  f1-score   support

              0       0.64      0.94      0.76        36
              1       0.89      0.46      0.60        35

       accuracy                           0.70        71
      macro avg       0.77      0.70      0.68        71
   weighted avg       0.76      0.70      0.69        71
   # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Best hyperparameters for SVM:
    {'C': 10, 'class_weight': {1: 1}, 'kernel': 'poly'}
    Best micro-averaged F1 score: 0.8964406779661017
    Accuracy: 0.704225352112676
    Confusion Matrix:
     [[34  2]
     [19 16]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.64      0.94      0.76        36
               1       0.89      0.46      0.60        35

        accuracy                           0.70        71
       macro avg       0.77      0.70      0.68        71
    weighted avg       0.76      0.70      0.69        71
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Best hyperparameters for Random Forest:
    {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 200}
    Best micro-averaged F1 score: 0.7793785310734462
    
    Accuracy: 0.704225352112676
    Confusion Matrix:
     [[34  2]
     [19 16]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.64      0.94      0.76        36
               1       0.89      0.46      0.60        35

        accuracy                           0.70        71
       macro avg       0.77      0.70      0.68        71
    weighted avg       0.76      0.70      0.69        71
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    Best hyperparameters for Logistic Regression:
    {'C': 0.01, 'penalty': 'l2'}
    Best micro-averaged F1 score: 0.8864406779661017
    Accuracy: 0.704225352112676
    Confusion Matrix:
     [[34  2]
     [19 16]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.64      0.94      0.76        36
               1       0.89      0.46      0.60        35

        accuracy                           0.70        71
       macro avg       0.77      0.70      0.68        71
    weighted avg       0.76      0.70      0.69        71
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Best hyperparameters for Decision Tree:
    {'max_depth': None, 'min_samples_split': 5}
    Best micro-averaged F1 score: 0.8893785310734466
    
    Accuracy: 0.6901408450704225
    Confusion Matrix:
     [[34  2]
     [20 15]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.63      0.94      0.76        36
               1       0.88      0.43      0.58        35

        accuracy                           0.69        71
       macro avg       0.76      0.69      0.67        71
    weighted avg       0.75      0.69      0.67        71
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    Best hyperparameters for KNN:
    {'n_neighbors': 7, 'p': 2, 'weights': 'distance'}
    Best micro-averaged F1 score: 0.8495480225988702
    Accuracy: 0.704225352112676
    Confusion Matrix:
     [[34  2]
     [19 16]]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.64      0.94      0.76        36
               1       0.89      0.46      0.60        35

        accuracy                           0.70        71
       macro avg       0.77      0.70      0.68        71
    weighted avg       0.76      0.70      0.69        71
    '''
   
    classifiers = {
    #'SVM': SVC(),
    #'Random Forest': RandomForestClassifier(),
    #'Logistic Regression': LogisticRegression(),
    #'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
     }

    param_grids = {
    'SVM': {'C': [0.01, 0.1, 1, 5, 10, 20, 100], 'kernel': ['rbf', 'poly'], 'class_weight': [{1: 1}]},
    'Random Forest': {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]},
    'Logistic Regression': {'C': [0.01, 0.1, 1, 5, 10], 'penalty': ['l1', 'l2']},
    'Decision Tree': {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]},
    'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'],'p': [1, 2]},
    
    }

    # Create an empty dictionary to store the best classifiers and their parameters
    best_classifiers = {}
    
    # Define a custom scoring function for micro-averaged F1
    scorer = make_scorer(f1_score, average='micro')

    # Perform hyperparameter tuning and comparison for each classifier
    for classifier_name, classifier in classifiers.items():
        param_grid = param_grids[classifier_name]
        grid = GridSearchCV(classifier, param_grid, cv=5, verbose=1, n_jobs=-1, scoring=scorer)
        grid.fit(training_data_vectors, training_classes)  
        print(f'Best hyperparameters for {classifier_name}:')
        best_params = grid.best_params_
        print(best_params)
        print(f'Best micro-averaged F1 score: {grid.best_score_}\n')
    
        # Store the best classifier and its parameters in the dictionary
        best_classifiers[classifier_name] = {
            'classifier': grid.best_estimator_,
            'best_params': best_params
            } 
      
    # Define the individual models with their best hyperparameters
    print('SVC')
    svm = SVC(C=10, class_weight={1: 1}, kernel='poly', probability=True)
    # Fit the ensemble model with your training data
    svm.fit(training_data_vectors, training_classes)
    y_pred = svm.predict(test_data_vectors)      
    #accuracy = accuracy_score(test_classes, y_pred)
    #conf_matrix = confusion_matrix(test_classes, y_pred)
    #report = classification_report(test_classes, y_pred)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)
    #print("Classification Report:\n", report)
    calculate_classification_metrics(test_classes ,y_pred)    
    
    print('RF')
    rf = RandomForestClassifier(max_depth=30, min_samples_split=5, n_estimators=200)
    rf.fit(training_data_vectors, training_classes)
    y_pred = rf.predict(test_data_vectors)      
    #accuracy = accuracy_score(test_classes, y_pred)
    #conf_matrix = confusion_matrix(test_classes, y_pred)
    #report = classification_report(test_classes, y_pred)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)
    #print("Classification Report:\n", report)
    calculate_classification_metrics(test_classes ,y_pred)
    
    print('LR')
    lr = LogisticRegression(C=0.01, penalty='l2')
    lr.fit(training_data_vectors, training_classes)
    y_pred = lr.predict(test_data_vectors)      
    #accuracy = accuracy_score(test_classes, y_pred)
    #conf_matrix = confusion_matrix(test_classes, y_pred)
    #report = classification_report(test_classes, y_pred)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)
    #print("Classification Report:\n", report)
    calculate_classification_metrics(test_classes ,y_pred)
    
    print('DT')
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=5)
    dt.fit(training_data_vectors, training_classes)
    y_pred = dt.predict(test_data_vectors)      
    #accuracy = accuracy_score(test_classes, y_pred)
    #conf_matrix = confusion_matrix(test_classes, y_pred)
    #report = classification_report(test_classes, y_pred)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)
    #print("Classification Report:\n", report)
    calculate_classification_metrics(test_classes ,y_pred)  
    
    print('knn')
    knn = KNeighborsClassifier(n_neighbors=7, p=2, weights='distance')
    knn.fit(training_data_vectors, training_classes)
    y_pred = knn.predict(test_data_vectors)      
    #accuracy = accuracy_score(test_classes, y_pred)
    #conf_matrix = confusion_matrix(test_classes, y_pred)
    #report = classification_report(test_classes, y_pred)
    #print("Accuracy:", accuracy)
    #print("Confusion Matrix:\n", conf_matrix)
    #print("Classification Report:\n", report)
    calculate_classification_metrics(test_classes ,y_pred) 
    
    '''
    Results of testing models after training with all the dataset
    
    SVC
    Accuracy: 0.8450704225352113
    Micro-F1: 0.8450704225352113
    Macro-F1: 0.8449473893190391
    Precision: [0.83783784 0.85294118]
    Recall: [0.86111111 0.82857143]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.84      0.86      0.85        36
               1       0.85      0.83      0.84        35

        accuracy                           0.85        71
       macro avg       0.85      0.84      0.84        71
    weighted avg       0.85      0.85      0.85        71

    RF
    Accuracy: 0.7464788732394366
    Micro-F1: 0.7464788732394367
    Macro-F1: 0.7439903846153846
    Precision: [0.71428571 0.79310345]
    Recall: [0.83333333 0.65714286]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.71      0.83      0.77        36
               1       0.79      0.66      0.72        35

        accuracy                           0.75        71
       macro avg       0.75      0.75      0.74        71
    weighted avg       0.75      0.75      0.74        71

    LR
    Accuracy: 0.8169014084507042
    Micro-F1: 0.8169014084507042
    Macro-F1: 0.816318407960199
    Precision: [0.79487179 0.84375   ]
    Recall: [0.86111111 0.77142857]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.79      0.86      0.83        36
               1       0.84      0.77      0.81        35

        accuracy                           0.82        71
       macro avg       0.82      0.82      0.82        71
    weighted avg       0.82      0.82      0.82        71

    DT
    Accuracy: 0.7746478873239436
    Micro-F1: 0.7746478873239436
    Macro-F1: 0.7742448330683624
    Precision: [0.76315789 0.78787879]
    Recall: [0.80555556 0.74285714]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.76      0.81      0.78        36
               1       0.79      0.74      0.76        35

        accuracy                           0.77        71
       macro avg       0.78      0.77      0.77        71
    weighted avg       0.78      0.77      0.77        71

    knn
    Accuracy: 0.9014084507042254
    Micro-F1: 0.9014084507042254
    Macro-F1: 0.9010945273631841
    Precision: [0.87179487 0.9375    ]
    Recall: [0.94444444 0.85714286]
    Classification Report:
                   precision    recall  f1-score   support

               0       0.87      0.94      0.91        36
               1       0.94      0.86      0.90        35

        accuracy                           0.90        71
       macro avg       0.90      0.90      0.90        71
    weighted avg       0.90      0.90      0.90        71
    '''
       
    # Create the VotingClassifier
    print('VotingClassifier')
    voting_classifier = VotingClassifier(estimators=[
        ('svm', svm),
        ('rf', rf),
        ('lr', lr),
        ('dt', dt),
        ('knn', knn)], voting='soft')  # You can choose 'hard' or 'soft' voting

    # Fit the ensemble model with your training data
    # voting_classifier.fit(training_data_vectors, training_classes)
    # y_pred = voting_classifier.predict(test_data_vectors)      
    # calculate_classification_metrics(test_classes ,y_pred)

