import spacy

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Example text
text = "The sun rose over the horizon, casting a warm glow across the fields. Birds chirped cheerfully, welcoming the new day. Farmers began their work, tending to the crops with care. By evening, the fields were filled with the scent of fresh earth and blooming flowers."

# Tokenization, Lemmatization, Stop Words Removal
doc = nlp(text)
tokens = [token.text for token in doc]
filtered_tokens = [token.text for token in doc if not token.is_stop]
stemmed_tokens = [token.lemma_ for token in doc]
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

print("Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
