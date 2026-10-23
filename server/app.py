from flask import Flask, request, render_template
import numpy as np

# extract the trained glove weights, context weights, and vocab dict.
glove_data = np.load("../glove/glove_model.npz", allow_pickle=True)
weights = glove_data["weights"]
context_weights = glove_data["context"]
vocab = glove_data["vocab"].item()

NUM_SIMILAR_WORDS = 10

app = Flask(__name__)

def get_closest_words_from_index(vocab_dict, weights, context_weights, index, n):
    # accept index==0 and None explicitly
    if index is None:
        return []

    # combine word and context embeddings
    word_vector = weights[index] + context_weights[index]
    word_vector = word_vector.astype(float)
    w_norm = np.linalg.norm(word_vector) + 1e-12

    similarities = []
    for word, i in vocab_dict.items():
        current_vector = weights[i] + context_weights[i]
        c_norm = np.linalg.norm(current_vector) + 1e-12
        cosine = np.dot(word_vector, current_vector) / (w_norm * c_norm)
        similarities.append({"word": word, "similarity": float(cosine)})

    similarities_sorted = sorted(similarities, key=lambda w: w["similarity"], reverse=True)
    return [w["word"] for w in similarities_sorted[:n]]



@app.route("/", methods=["GET"])
def index():
    word = request.args.get("word")
    similar_words = []

    if word:
        word_index = vocab.get(word)
        # use the vocab mapping (word -> index) when querying nearest neighbors
        similar_words = get_closest_words_from_index(
            vocab,
            weights, 
            context_weights,
            word_index, 
            NUM_SIMILAR_WORDS
        )

    return render_template("index.html", word=word, similar_words=similar_words)
