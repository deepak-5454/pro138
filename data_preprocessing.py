# Text Data Preprocessing Lib
import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Create an instance of PorterStemmer
stemmer = PorterStemmer()

# List of words to be ignored while creating the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

# Lists to store unique root words and tags
words = []
classes = []
pattern_word_tags_list = []

# Open and read data from the JSON file
with open('intents.json') as train_data_file:
    data = json.load(train_data_file)

# Function to stem words
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

# Function to create the bot corpus
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        # Add all patterns and tags to a list
        for pattern in intent['patterns']:
            pattern_word = nltk.word_tokenize(pattern)
            words.extend(pattern_word)
            pattern_word_tags_list.append((pattern_word, intent['tag']))

        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    # Get unique stemmed words and tags
    stem_words = get_stem_words(words, ignore_words)
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list

# Function to perform Bag of Words encoding
def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0]
        bag_of_words = [1 if word in get_stem_words(pattern_words, ignore_words) else 0 for word in stem_words]
        bag.append(bag_of_words)
    return np.array(bag)

# Function to perform class label encoding
def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        labels_encoding = [0] * len(classes)
        tag = word_tags[1]
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

# Function to preprocess training data
def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    
    # Convert stem words and classes to Python pickle file format
    pickle.dump(stem_words, open('words.pkl', 'wb'))
    pickle.dump(tag_classes, open('classes.pkl', 'wb'))

    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    
    return train_x, train_y

# Preprocess the data and print the first Bag of Words encoding and label encoding
bow_data, label_data = preprocess_train_data()
print("First Bag of Words encoding:", bow_data[0])
print("First Label encoding:", label_data[0])
