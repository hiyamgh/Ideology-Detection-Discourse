import pickle
import argparse
import numpy as np
from transformers import AutoTokenizer


def get_cos_dist(words, path, years):

    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f)

    for w in words:

        for year in years:
            year_word = w + '_' + year
            if year_word in vocab_vectors:
                print(vocab_vectors[year_word])
                print()
            else:
                print(f'Could not find {year_word}')
                tokenized_word = tokenizer.tokenize(w)

                if not tokenized_word:
                    print(f"The word '{oov_word}' could not be tokenized.")
                    continue

                # Check if any of the tokens are in the vocab_vectors
                embeddings = []
                for token in tokenized_word:
                    # Check if the token exists in the vocab_vectors
                    token_key = token.replace('##', '')  # Remove '##' if it exists for subwords
                    if f"{token_key}_{year}" in vocab_vectors:  # Add the year if your embeddings are year-specific
                        embeddings.append(vocab_vectors[f"{token_key}_{year}"])

                if embeddings:
                    # Average the embeddings of the subwords
                    aggregated_embedding = np.mean(embeddings, axis=0)
                    print(aggregated_embedding)
                else:
                    print(f"No embeddings found for any tokens of the word '{oov_word}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=str,
                        help='Path to output time embeddings',
                        default='embeddings/liverpool.pickle')
    parser.add_argument('--shifts_path', type=str,
                        help='Path to gold standard semantic shifts path',
                        default='data/liverpool/liverpool_shift.csv')
    args = parser.parse_args()

    years = ['06', '07', '08', '09', '10', '11', '12']

    words = ['اسرائيل', 'فلسطين', 'ياسر عرفات', 'عرفات']

    path = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/embeddings.pickle'
    # Load the tokenizer used for generating the embeddings
    tokenizer = AutoTokenizer.from_pretrained('/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/UBC-NLP-MARBERTv2/')
    get_cos_dist(words=words, path=path, years=years)

