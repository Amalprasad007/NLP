import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import stem_text

# Example text
text = "The sun rose over the horizon, casting a warm glow across the fields. Birds chirped cheerfully, welcoming the new day. Farmers began their work, tending to the crops with care. By evening, the fields were filled with the scent of fresh earth and blooming flowers."

# Tokenization and stop words removal using Gensim
tokens = simple_preprocess(text, deacc=True)
filtered_tokens = [token for token in tokens if token not in STOPWORDS]
print("Tokens (Gensim):", tokens)
print("Filtered Tokens (Gensim):", filtered_tokens)

# Stemming using Gensim
stemmed_tokens = [stem_text(token) for token in filtered_tokens]
print("Stemmed Tokens (Gensim):", stemmed_tokens)
