import re
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import itertools
import Levenshtein
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
from collections import Counter

def run_sliding_window_through_text(words, window_size):
    """
    Generate a window sliding through a sequence of words
    """
    word_iterator = iter(words)  # creates an object which can be iterated one element at a time
    word_window = tuple(itertools.islice(word_iterator, window_size))  # islice() makes an iterator that returns selected elements from the word_iterator
    yield word_window

    # Now to move the window forward, one word at a time
    for w in word_iterator:
        word_window = word_window[1:] + (w,)
        yield word_window

def match_dict_similarity(text, expressions):
    
    threshold = 0.8
    max_similarity_obtained = -1
    best_match = ''

    # Go through each expression
    for exp in expressions:
        # Create the window size equal to the number of words in the expression in the lexicon
        size_of_window = len(exp.split())

        tokenized_text = list(nltk.word_tokenize(text))
        for window in run_sliding_window_through_text(tokenized_text, size_of_window):
            window_string = ' '.join(window)
            similarity_score = Levenshtein.ratio(window_string, exp)
            if similarity_score >= threshold:
                print(similarity_score, '\t', exp, '\t', window_string)
            if similarity_score > max_similarity_obtained:
                max_similarity_obtained = similarity_score
                best_match = window_string
    #print(best_match, max_similarity_obtained)
    return best_match

def in_scope(neg_end, text,symptom_expression):
    '''
    Function to check if a symptom occurs within the scope of a negation based on some
    pre-defined rules.
    :param neg_end: the end index of the negation expression
    :param text:
    :param symptom_expression:
    :return:
    '''
    negated = False
    text_following_negation = text[neg_end:]
    tokenized_text_following_negation = list(nltk.word_tokenize(text_following_negation))
    # this is the maximum scope of the negation, unless there is a '.' or another negation
    three_terms_following_negation = ' '.join(tokenized_text_following_negation[:min(len(tokenized_text_following_negation),3)])
    #Note: in the above we have to make sure that the text actually contains 3 words after the negation
    #that's why we are using the min function -- it will be the minimum or 3 or whatever number of terms are occurring after
    #the negation. Uncomment the print function to see these texts.
    #print (three_terms_following_negation)
    match_object = re.search(symptom_expression,three_terms_following_negation)
    if match_object:
        period_check = re.search('\.',three_terms_following_negation)
        next_negation = 1000 #starting with a very large number
        #searching for more negations that may be occurring
        for neg in negations:
            # a little simplified search..
            if re.search(neg,text_following_negation):
                index = text_following_negation.find(neg)
                if index<next_negation:
                    next_negation = index
        if period_check:
            #if the period occurs after the symptom expression
            if period_check.start() > match_object.start() and next_negation > match_object.start():
                negated = True
        else:
            negated = True
    return negated

def tokenize_text(text):
    # Split the text into words using whitespace as the delimiter
    return text.split()

def negation(text_tokens, negations):
    for token in text_tokens:
        if token in negations:
            return 1
    return 0

symptom_dict = {}
ICU_dict={}
infile = open('./COVID-Twitter-Symptom-Lexicon.txt')
for line in infile:
    items = line.split('\t')
    symptom_dict[str.strip(items[-1].lower())] = str.strip(items[1])
    ICU_dict[str.strip(items[0].lower())] = str.strip(items[1])
    
#loading the negation expressions
negations = []
infile = open('./neg_trigs.txt')
for line in infile:
    negations.append(str.strip(line))

# Specify the path to your Excel file
excel_file_path = 'Assignment1GoldStandardSet.xlsx'
excel_file_path ='UnlabeledSet.xlsx'

# Read the Excel file into a pandas DataFrame using the 'openpyxl' engine
df = pd.read_excel(excel_file_path)

# Access and manipulate the data as needed
required_columns = ['ID', 'TEXT', 'Symptom CUIs', 'Negation Flag']

df_cui_output=[]
df_negation_output=[]
df_ID=[]
df_text=[]
df_CUI=[]


if all(column in df.columns for column in required_columns):
    
    # Loop through the file names
    for index, row in df.iterrows(): 
        
        cui_values = []
        negation_values = []
        cui_values_dict = {}  
        text = str(row['TEXT']).lower()
        sentences = sent_tokenize(text)
        #so that it can be processed later by a negation scoping function
        matched_tuples1 = []
        matched_tuples2 = []
        #go through each sentence
        for s in sentences:
            #go through each symptom expression in the dictionary
            for symptom in symptom_dict.keys():
                #find all matches
                for match in re.finditer(r'\b'+symptom+r'\b',s):
                    match_tuple = (s,symptom_dict[symptom],match.group(),match.start(),match.end())
                    matched_tuples1.append(match_tuple)
                    best_match = match_dict_similarity(s,symptom_dict[symptom])
                    best_match =(s, best_match, symptom_dict[symptom])
                    matched_tuples2.append(best_match)  
        
        #now to check if a concept is negated or not
        for mt in matched_tuples1:
            is_negated = False
            #Note: I broke down the code into simpler chunks for easier understanding..
            text = mt[0]
            cui = mt[1]
            expression = mt[2]
            #start = mt[3]
            end = mt[4]
            cui_values_dict[cui] = True  # Mark CUI as added
            #end = 895
            #uncomment the print calls to separate each text fragment..
            #print('=------=')

            # Go through each negation expression
            for neg in negations:
                    for match in re.finditer(r'\b'+neg+r'\b', text):
                        is_negated = in_scope(match.end(), text, expression)
                        if is_negated:
                            cui_values.append(f"{cui}")
                            cui_values_dict[cui]=1
                            negation_values.append("1")
                            break
                
            if not is_negated:
                    cui_values_dict[cui]=0
                    cui_values.append(f"{cui}")
                    negation_values.append("0")
                    
        # now to check if a concept is negated or not 
        for mt in matched_tuples2:
             # Initialize is_negated to False for each matched tuple
             is_negated = False
             text = mt[0]
             cui = mt[2]

             # Tokenize the text
             text_tokens = tokenize_text(text)
             # Check if any token in the text matches a negation expression
           
             negated_result = negation(text_tokens, negations)  # Assign the result to a different variable
             
             if cui not in cui_values_dict:

                 if negated_result==1:  # Use the new variable to check if it's negated
                     cui_values_dict[cui] = True
                     cui_values.append(f"{cui}")
                     negation_values.append("1")
                 else:
                    cui_values_dict[cui] = True
                    cui_values.append(f"{cui}")
                    negation_values.append("0")
                    
             if cui in cui_values_dict: 
                 if negated_result==1:  # Use the new variable to check if it's negated
                     if cui_values_dict[cui] ==0:
                        cui_values_dict[cui] = True
                        cui_values.append(f"{cui}")
                        negation_values.append("1")
                 if negated_result==0:  # Use the new variable to check if it's negated
                     if cui_values_dict[cui] ==1:
                        cui_values_dict[cui] = True
                        cui_values.append(f"{cui}")
                        negation_values.append("0")
                        
        # Print the cui_values and negation_values separated by $$$
        cui_output = "$$$" + "$$$".join(cui_values) + "$$$"
        negation_output = "$$$" + "$$$".join(negation_values) + "$$$"
        df_CUI.append(cui_values)
        df_ID.append(row['ID'])
        df_text.append(row['TEXT'])
        df_cui_output.append(cui_output)
        df_negation_output.append(negation_output)
        print(cui_output)
        print(negation_output)           

# Create a DataFrame with the collected lists
result_df = pd.DataFrame({'ID': df_ID,'TEXT':df_text,'Symptom CUIs': df_cui_output,'Negation Flag': df_negation_output})

# Save the DataFrame as an Excel file
result_df.to_excel('result.xlsx')
result_df.to_excel('result_UnlabeledSet.xlsx')


# Plot Histogram
# Flatten the list of lists (df_CUI) into a single list
flattened_cui = [item for sublist in df_CUI for item in sublist]
# Count the occurrences of each symptom in the flattened list
symptom_counts = Counter(flattened_cui)
# Get the 10 most common symptoms
most_common_symptoms = symptom_counts.most_common(10)
# Extract the symptom names and their corresponding counts
symptoms, counts = zip(*most_common_symptoms)
# Create a bar plot (histogram) for the top 10 symptoms
plt.figure(figsize=(12, 6))
plt.bar(symptoms, counts)
plt.xlabel('Symptoms')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Symptoms')
# Rotate x-axis labels for better readability if needed
plt.xticks(rotation=45)  
# Show the plot
plt.savefig('top_10_symptoms.png', dpi=300)
plt.tight_layout()
plt.show()