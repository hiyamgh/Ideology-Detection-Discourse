import torch
import os
from helper import *
import pickle
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from normalization import ArabicNormalizer
from tqdm import tqdm


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_word_embedding_static(word, tokenizer, model):
    tokenized = tokenizer(word, return_tensors="pt", add_special_tokens=True)

    # Tokenized input
    input_ids = tokenized["input_ids"].to(device)  # Token IDs
    attention_mask = tokenized["attention_mask"].to(device)  # Attention mask

    # Get hidden states from BERT
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Shape: (batch_size, sequence_length, hidden_size)

    # Extract the embedding for the word
    # Note: Skip [CLS] and [SEP] tokens
    token_embeddings = hidden_states[0, 1:-1, :]  # Shape: (num_tokens, hidden_size)

    # If the word is split into subwords, aggregate embeddings
    word_embedding = token_embeddings.mean(dim=0)  # Take the mean of subword embeddings
    print(f"Word embedding shape of {word}: {word_embedding.shape}")
    return word_embedding


def get_vector_from_similar_words(word_token, year, embeddings, tokenizer):
    with open("../generate_bert_embeddings/similarities.pkl", "rb") as file:
        loaded_dict = pickle.load(file)

    try:
        similar_words = loaded_dict[word_token]
    except:
        return []
    print(f'similar words to {word_token}: {similar_words}')
    vectors = []
    for w in similar_words:
        v = get_vector(w, year, embeddings, tokenizer)
        vectors.append(v)

    if vectors:
        vectors = [v for v in vectors if isinstance(v, np.ndarray)]
        embedding = np.mean(vectors, axis=0)
        return embedding
    else:
        return []


def get_vector(word_token, year, embeddings, tokenizer):
    arabnormalizer = ArabicNormalizer()
    if '{}_{}'.format(word_token, year) in embeddings:
        emb_temp = embeddings['{}_{}'.format(word_token, year)]
        return emb_temp
    else:
        all_tokens_found = True
        vectors = []
        tokenized_word = tokenizer.tokenize(word_token)
        if not tokenized_word:
            print(f"The word '{word_token}' could not be tokenized.")
            return -1

        for token in tokenized_word:
            token_key = token
            if f"{token_key}_{year}" in embeddings:  # Add the year if your embeddings are year-specific
                vectors.append(embeddings[f"{token_key}_{year}"])
            else:
                token_key = token_key.replace("##", "")
                if f"{token_key}_{year}" in embeddings:  # Add the year if your embeddings are year-specific
                    vectors.append(embeddings[f"{token_key}_{year}"])
                else:
                    t_new = arabnormalizer.normalize_token(token=token_key)
                    if f"{t_new}_{year}" in embeddings:  # Add the year if your embeddings are year-specific
                        vectors.append(embeddings[f"{t_new}_{year}"])
                    else:
                        print(f"{token_key}_{year} IS NOT IN EMBEDDINGS.")
                        all_tokens_found = False

        if not all_tokens_found:
            return []
        else:
            embedding = np.mean(vectors, axis=0)
            return embedding


def get_time_specific_word_embedding(word, year, embeddings, tokenizer, model):
    if '{}_{}'.format(word, year) in embeddings:
        embedding = embeddings['{}_{}'.format(word, year)]
        # return embedding, count
        return embedding
    else:
        vectors = []
        if len(word.split(" ")) > 1:
            for wt in word.split(" "):
                wt = wt.strip()
                v = get_vector(word_token=wt, year=year, embeddings=embeddings, tokenizer=tokenizer)
                if isinstance(v, np.ndarray):
                    vectors.append(v)
                else:
                    vupdated = get_vector_from_similar_words(word_token=wt, year=year, embeddings=embeddings, tokenizer=tokenizer)
                    try:
                        print(f'vupdated shape: {vupdated.shape}')
                        if vupdated.shape == ():
                            vupdated = get_word_embedding_static(word=wt, tokenizer=tokenizer, model=model)
                            # count += 1
                    except:
                        vupdated = get_word_embedding_static(word=wt, tokenizer=tokenizer, model=model)
                        # count += 1
                    vectors.append(vupdated)

        else:
            word = word.strip()
            v = get_vector(word_token=word, year=year, embeddings=embeddings, tokenizer=tokenizer)
            if isinstance(v, np.ndarray):
                vectors.append(v)
            else:
                vupdated = get_vector_from_similar_words(word_token=word, year=year, embeddings=embeddings, tokenizer=tokenizer)
                try:
                    print(f'vupdated shape: {vupdated.shape}')
                    if vupdated.shape == ():
                        vupdated = get_word_embedding_static(word=word, tokenizer=tokenizer, model=model)
                        # count += 1
                except:
                    vupdated = get_word_embedding_static(word=word, tokenizer=tokenizer, model=model)
                    # count += 1
                vectors.append(vupdated)

        vectors = [v for v in vectors if isinstance(v, np.ndarray)]
        if vectors:
            embedding = np.mean(vectors, axis=0)
        else:
            embedding = get_word_embedding_static(word=word, tokenizer=tokenizer, model=model)

        # return embedding, count
        return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_name", default="An-Nahar", help="name of the archive to compute embeddings for")
    parser.add_argument("--model_name", default="UBC-NLP-MARBERT", help="model name for archive")
    args = parser.parse_args()

    archive_name = args.archive_name
    model_name = args.model_name

    path = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/{}/{}/'.format(archive_name, model_name)
    path_to_model = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{}/{}/".format(archive_name, model_name)

    # tokenizer for the archive
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    # models for each archive
    model = AutoModelForMaskedLM.from_pretrained(path_to_model, output_hidden_states=True).cuda()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(path, 'embeddings.pickle'), 'rb') as handle:
        embeddings = pickle.load(handle)
        print(len(embeddings))

    vocab_words = list(tokenizer.get_vocab().keys())

    years = ['06', '07', '08', '09', '10', '11', '12']
    vocabs_to_embeddings = {}

    save_dir = f"vocabulary_embeddings/{archive_name}/{model_name}/"
    mkdir(save_dir)

    for v in tqdm(vocab_words):
        for year in years:
            v_emb_y = get_time_specific_word_embedding(word=v, year=year, embeddings=embeddings, tokenizer=tokenizer, model=model)

            if isinstance(v_emb_y, np.ndarray): # if numpy array make it a tensor
                v_emb_y = torch.tensor(v_emb_y).to(device)
            if v_emb_y.shape != (1, 768):       # ensure all vectors are of shape (1, 768)
                v_emb_y = v_emb_y.unsqueeze(0).to(device)

            vocabs_to_embeddings[f"{v}_{year}"] = v_emb_y

            with open(os.path.join(save_dir, f'{archive_name}_{model_name}.pickle'), 'wb') as handle:
                pickle.dump(vocabs_to_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

