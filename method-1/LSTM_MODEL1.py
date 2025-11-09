import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = load_model('models/Model1.h5')


# Function for cleaning text
def text_cleaner(tx):
    text = re.sub(r"won't", "would not", tx)
    text = re.sub(r"im", "i am", tx)
    text = re.sub(r"Im", "I am", tx)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"needn't", "need not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"mightn't", "might not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\!\?\.\@]',' ' , text)
    text = re.sub(r'[!]+' , '!' , text)
    text = re.sub(r'[?]+' , '?' , text)
    text = re.sub(r'[.]+' , '.' , text)
    text = re.sub(r'[@]+' , '@' , text)
    text = re.sub(r'unk' , ' ' , text)
    text = re.sub('\n', '', text)
    text = text.lower()
    text = re.sub(r'[ ]+' , ' ' , text)
    return text

# Read the CSV file
df = pd.read_csv('data/youtube_comments.csv')

df['text'] = df['text'].astype(str)


# Preprocess the text data
df['cleaned_text'] = df['text'].apply(text_cleaner)

# Tokenize and pad the sequences
max_sequence_length = 100  # Define maximum sequence length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
X = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(X, maxlen=max_sequence_length)

# Predict the sentiments using the saved model
predicted_probabilities = model.predict(X)

# Convert probabilities to sentiment labels
sentiment_labels = [ 'Negative','Positive', 'Neutral']
predicted_sentiments = [sentiment_labels[np.argmax(prob)] for prob in predicted_probabilities]

# Count the occurrences of each sentiment class
sentiment_counts = pd.Series(predicted_sentiments).value_counts()

# Print the counts
print("Sentiment Counts:")
print(sentiment_counts)
# Calculate percentages of each sentiment class
total_samples = len(df)
sentiment_percentages = sentiment_counts / total_samples * 100

# Print the percentages
print("Sentiment Percentages:")
print(sentiment_percentages)
import matplotlib.pyplot as plt

# Sentiment percentages from previous calculation
sizes = sentiment_percentages.values
labels = sentiment_percentages.index
colors = ['lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0)  # explode 1st slice (Positive)

# Plot pie chart for sentiment distribution
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


