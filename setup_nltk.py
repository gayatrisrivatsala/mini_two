import nltk

# This script downloads the 'punkt' tokenizer models, needed for splitting text into sentences.
# You only need to run this once.
print("Downloading NLTK 'punkt' tokenizer data...")
nltk.download('punkt')
print("Download complete.")