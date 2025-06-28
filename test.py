import pickle
from embedding.FastText import FastTextLSAEmbedder, FastText, TruncatedSVD


model_name = "fasttext"

with open(f"./trained_models/{model_name}.pkl", 'rb') as f:
        print(model_name)
        embedder = pickle.load(f)

# print(embedder.keys())
print(dir(embedder))

# print(type(embedder))
# print(embedder)

