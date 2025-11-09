import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Function for cleaning text
def text_cleaner(tx):
    if not isinstance(tx, str):
        return ''
    
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

# Function to extract emojis from text
def extract_emojis(text):
    if not isinstance(text, str):
        return ''
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return ''.join(emoji_pattern.findall(text))


# Function to retrieve emoji scores
def get_emoji_scores(emojis, emoji_scores_df):
    emoji_list = list(emojis)
    neg_scores = []
    neut_scores = []
    pos_scores = []
    if emoji_list:
        for emoji in emoji_list:
            emoji_score_row = emoji_scores_df[emoji_scores_df['Char'] == emoji]
            if not emoji_score_row.empty:
                neg_score = emoji_score_row.iloc[0]['Neg']
                neut_score = emoji_score_row.iloc[0]['Neut']
                pos_score = emoji_score_row.iloc[0]['Pos']
            else:
                neg_score = 0
                neut_score = 0
                pos_score = 0
            neg_scores.append(neg_score)
            neut_scores.append(neut_score)
            pos_scores.append(pos_score)
    else:
        neg_scores.append(0)
        neut_scores.append(0)
        pos_scores.append(0)
    avg_neg_score = sum(pd.to_numeric(neg_scores)) / len(neg_scores)
    avg_neut_score = sum(pd.to_numeric(neut_scores)) / len(neut_scores)
    avg_pos_score = sum(pd.to_numeric(pos_scores)) / len(pos_scores)
    return avg_neg_score, avg_neut_score, avg_pos_score

# Read the CSV file containing emoji polarity scores
emoji_scores_df = pd.read_csv('D:\Major_final\code\method-2\ijstable.csv')

# Read the CSV file
df = pd.read_csv('data/youtube_comments.csv')

# Preprocess the text data
df['cleaned_text'] = df['text'].apply(text_cleaner)

# Tokenize and pad the sequences
max_sequence_length = 100  # Define maximum sequence length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_text'])
X = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(X, maxlen=max_sequence_length)

# Load the trained model
model = load_model('models/Model2.h5')



# Extract emojis from text
df['emojis'] = df['text'].apply(extract_emojis)


# Retrieve emoji scores
df['emoji_neg'], df['emoji_neut'], df['emoji_pos'] = zip(*df['emojis'].apply(lambda emojis: get_emoji_scores(emojis, emoji_scores_df)))

# Predict the sentiments using the saved model
predicted_probabilities = model.predict(X)

# Extract positive, negative, and neutral scores
positive_score = predicted_probabilities[:, 2]  # Probability for 'Positive' sentiment
negative_score = predicted_probabilities[:, 0]  # Probability for 'Negative' sentiment
neutral_score = predicted_probabilities[:, 1]   # Probability for 'Neutral' sentiment

# Combine text and emoji scores
total_pos = 0.5 * positive_score + 0.5 * df['emoji_pos']
total_neg = 0.5 * negative_score + 0.5 * df['emoji_neg']
total_neut = 0.5 * neutral_score + 0.5 * df['emoji_neut']

# Determine sentiment labels
sentiment_labels = ['Negative', 'Positive', 'Neutral']
df['predicted_sentiment'] = np.array(sentiment_labels)[np.argmax(predicted_probabilities, axis=1)]

# Calculate sentiment counts and percentages
sentiment_counts = df['predicted_sentiment'].value_counts()
total_comments = len(df)
pos_percentage = (sentiment_counts['Positive'] / total_comments) * 100
neg_percentage = (sentiment_counts['Negative'] / total_comments) * 100
neut_percentage = (sentiment_counts['Neutral'] / total_comments) * 100

# Print sentiment statistics
print("Sentiment Counts:")
print(sentiment_counts)
print("Total comments:", total_comments)
print("Positive percentage:", pos_percentage)
print("Negative percentage:", neg_percentage)
print("Neutral percentage:", neut_percentage)
import matplotlib.pyplot as plt

# Data for the pie chart
sizes = [pos_percentage, neg_percentage, neut_percentage]
labels = ['Positive', 'Negative', 'Neutral']
colors = ['lightcoral', 'lightskyblue', 'lightgreen']
explode = (0.1, 0, 0)  # explode the 1st slice (Positive)

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
