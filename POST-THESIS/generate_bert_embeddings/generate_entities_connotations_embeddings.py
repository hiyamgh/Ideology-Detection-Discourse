import pickle
import argparse
import os
import numpy as np
from transformers import AutoTokenizer


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_embeddings_dictionary(words, path, years):

    with open(path, 'rb') as f:
        vocab_vectors = pickle.load(f)

    final_embeddings = {}

    for w in words:

        for year in years:
            year_word = w + '_' + year
            if year_word in vocab_vectors:
                print(vocab_vectors[year_word].shape)
                final_embeddings[year_word] = vocab_vectors[year_word]
            else:
                print(f'Could not find {year_word}')
                tokenized_word = tokenizer.tokenize(w)

                if not tokenized_word:
                    print(f"The word '{w}' could not be tokenized.")
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
                    print(aggregated_embedding.shape)
                    final_embeddings[year_word] = aggregated_embedding
                else:
                    print(f"No embeddings found for any tokens of the word '{w}'.")

    return final_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_name', type=str, help='name of the archive for which embeddings are stored')
    parser.add_argument('--model_name', type=str, help='name of the model for which embeddings are stored')
    args = parser.parse_args()

    years = ['06', '07', '08', '09', '10', '11', '12']

    words = []
    for file in os.listdir('entities/bias_quantification/'):
        with open(os.path.join('entities/bias_quantification/', file), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                w = line.replace('\n', '').strip()
                print(w)
                words.append(w)

    for file in os.listdir('entities/contrastive_summaries/'):
        with open(os.path.join('entities/contrastive_summaries/', file), encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                w = line.replace('\n', '').strip()
                print(w)
                words.append(w)

    path = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/{}/{}/embeddings.pickle'.format(
        args.archive_name, args.model_name
    )
    tokenizer = AutoTokenizer.from_pretrained('/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{}/'.format(args.model_name))
    final_embeddings = get_embeddings_dictionary(words=words, path=path, years=years)
    save_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/{}/{}/'.format(args.archive_name, args.model_name)
    mkdir(save_dir)
    with open(os.path.join(save_dir, 'words_per_year.pickle'), 'wb') as handle:
        pickle.dump(final_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


