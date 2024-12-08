import torch
import os
import numpy as np
import pickle
import argparse
import matplotlib
matplotlib.use('Agg')
from transformers import AutoTokenizer, AutoModelForMaskedLM
from normalization import ArabicNormalizer
from annoy import AnnoyIndex
from scipy.linalg import orthogonal_procrustes
from datetime import date
from tqdm import tqdm


def get_word_embedding_static(word, tokenizer, model):
    """ Get static word embedding for a single word """
    tokenized = tokenizer(word, return_tensors="pt", add_special_tokens=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """ Sin
        Due to the presence of OCR errors, we are not finding the time-specific token representations,
        so we resort to getting the embeddings of the words that are very similar to the original word,
        which was originally measured through jaccard similarity with a threshold >= 0.70
    """
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
    """ Get the word/token embedding """
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
    """ get time specific word embedding """
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


def mkdir(folder):
    """ creates a directory if it doesn't already exist """
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_cosine_sim(v1, v2):
    """ Get the cosine similarity between two vectors """
    v1 = v1.flatten()  # Convert to 1D if necessary
    v2 = v2.flatten()  # Convert to 1D if necessary

    try:
        v1 = v1.cpu().numpy()
    except:
        pass

    try:
        v2 = v2.cpu().numpy()
    except:
        pass

    # return dot(v1, v2) / (norm(v1) * norm(v2))
    print(f'cos sim: {np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))}')
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def load_linear_mapping_matrices(dir_name, model1_name, model2_name):
    """
    load the linear transformation matrices between one embedding and another
    :param dir_name: name of the directory where the matrices are stored
    :param mat_name: file name used to store the matrices (this is a suffix; the
                    inverse transformation matrix has an extra substring '_inv')
    :return the transformation matrix R and its inverse R_inv
    """
    R = np.load(os.path.join(dir_name, '{}_{}.npy').format(model1_name, model2_name))
    R_inv = np.load(os.path.join(dir_name, '{}_{}_inv.npy'.format(model1_name, model2_name)))
    return R, R_inv


def get_stability_linear_mapping_one_word(model1, model2, embeddings_1, embeddings_2, tokenizer_1, tokenizer_2, model1_name, model2_name, mat_name, w):
    ''' gets the stability values of one word only
        assumes that transformation matrices are already computed
    '''
    W_i_j, W_j_i = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=model1_name, model2_name=model2_name)
    V_i_w = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_1, tokenizer=tokenizer_1, model=model1)
    V_j_w = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_2, tokenizer=tokenizer_2, model=model2)

    print(f'W_i_j.shape: {W_i_j.shape}, W_j_i.shape: {W_j_i.shape},')
    print(f'V_i_w.shape: {V_i_w.shape}, V_j_w.shape: {V_j_w.shape}')
    if V_i_w.shape != (1, 768):
        V_i_w = V_i_w.reshape(1, -1)
    if V_j_w.shape != (1, 768):
        V_j_w = V_j_w.reshape(1, -1)

    return compute_stability(V_i_w, V_j_w, W_i_j, W_j_i)


def compute_stability(V_i_w, V_j_w, W_i_j, W_j_i):
    if np.isnan(W_i_j).any() or np.isnan(W_j_i).any():
        print("NaN found in transformation matrices!")

    # Ensure vectors are 1D
    V_i_w = V_i_w.flatten()  # Convert to 1D if necessary
    V_j_w = V_j_w.flatten()  # Convert to 1D if necessary

    # Reshape V_i_w to (768,) and V_j_w to (768,) for correct multiplication
    V_i_w = V_i_w.reshape(768, )
    V_j_w = V_j_w.reshape(768, )

    try:
        V_i_w = V_i_w.cpu().numpy()
    except:
        pass
    try:
        V_j_w = V_j_w.cpu().numpy()
    except:
        pass
    try:
        W_i_j = W_i_j.cpu().numpy()
    except:
        pass

    try:
        W_j_i = W_j_i.cpu().numpy()
    except:
        pass

    # Forward-backward mappings
    mapped_forward = W_j_i.dot(W_i_j.dot(V_i_w))  # W^{j,i}W^{i,j}V^i_w
    mapped_backward = W_i_j.dot(W_j_i.dot(V_j_w))  # W^{i,j}W^{j,i}V^j_w

    # Reshape V_j_w correctly to ensure compatibility with W_j_i
    # mapped_backward = W_i_j @ (W_j_i @ V_j_w.reshape(338, 1))  # W^{i,j}W^{j,i}V^j_w

    print(f'mapped_forward.shape: {mapped_forward.shape}')
    print(f'mapped_backward.shape: {mapped_backward.shape}')

    # Cosine similarities
    sim_01 = get_cosine_sim(mapped_forward.reshape(1, -1), V_i_w.reshape(1, -1))
    sim_10 = get_cosine_sim(mapped_backward.reshape(1, -1), V_j_w.reshape(1, -1))

    # Stability measure
    stability = (sim_01 + sim_10) / 2
    return stability


def learn_stability_matrices(model1, model2, embeddings_1, embeddings_2, tokenizer_1, tokenizer_2, model1_name, model2_name, subwords, save_dir):
    """
    Get the stability by applying a transformation matrix that maps word w in embedding space 1
    to word w in embedding space 2. Transformation matrix is learned through gradient descent optimization
    by applying Forbenious norm in order to minimize loss. It uses a set of training words (stopwords or
    any set of frequent words that have a stable meaning regardless of time/source)

    :param model1: trained model - embedding 1
    :param model2: trained model - embedding 2
    :param subwords: list of words/stopwords to consider for learning the transformation matrix
    :return:
    """
    # print('Getting linear stability for the word {}'.format(w))

    def get_matrices():
        # create the matrices X and Y of source embeddings i and target embeddings j
        X, Y = [], []

        for w in subwords:
            x = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_1, tokenizer=tokenizer_1, model=model1)
            y = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_2, tokenizer=tokenizer_2, model=model2)

            X.append(x)
            Y.append(y)

        X = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in X]
        X = np.vstack(X)
        Y = [y.cpu().numpy() if isinstance(y, torch.Tensor) else y for y in Y]
        Y = np.vstack(Y)

        return X, Y

    def save_matrices(R, R_inv):
        # save transformation matrix and its inverse
        np.save(os.path.join(save_dir, '{}_{}.npy'.format(model1_name, model2_name)), R)
        np.save(os.path.join(save_dir, '{}_{}_inv.npy'.format(model1_name, model2_name)), R_inv)

    if os.path.exists(os.path.join(save_dir, '{}_{}.npy'.format(model1_name, model2_name))):
        print('matrices found in {}'.format(os.path.join(save_dir, '{}_{}.npy'.format(model1_name, model2_name))))
        return

    print('Calculating stability values (linear)')
    V_i, V_j = get_matrices()
    assert V_i.shape == V_j.shape

    W_i_j, _ = orthogonal_procrustes(V_i, V_j) # Perform Orthogonal Procrustes
    print(f'W_i_j.shape: {W_i_j.shape}')
    W_j_i = np.linalg.inv(W_i_j) # Compute W_j_i as the inverse of W_i_j
    print(W_j_i)
    print('got transformation matrices ...')

    save_matrices(R=W_i_j, R_inv=W_j_i)
    print('saved transformation matrices ...')


def get_stability_linear_mapping(model1, model2, embeddings_1, embeddings_2,
                                 tokenizer_1, tokenizer_2, model1_name, model2_name,
                                 mat_name, words_path, save_dir='results/',
                                 file_name='stabilities_linear'):

    print('Calculating stability values (linear)')

    with open(words_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]  # remove '\n'
    words = [w for w in words if w.strip() != '']
    vocab = [w for w in words]

    print('len of vocab: {}'.format(len(vocab)))
    # dictionary mapping each word w to its stability value. Before any processing, stability of any word w is 1.0
    stabilities = {}
    for w in vocab:
        stabilities[w] = 1.0

    for w in vocab:
        stability = get_stability_linear_mapping_one_word(model1, model2,
                                                                 embeddings_1, embeddings_2,
                                                                 tokenizer_1, tokenizer_2,
                                                                 model1_name, model2_name,
                                                                 mat_name, w)

        stabilities[w] = stability

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_combined_one_word(models,
                           embeddings,
                           tokenizers,
                           models_names,
                           annoy_indexes,
                           vocab_names,
                           dir_name_matrices,
                           word,
                           year,
                           k=50):
    print('Calculating stability values (combined) - k={}'.format(k))
    similarities = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                nnsims1 = get_nearest_neighbors(word=word, year=year, embeddings=embeddings[i], tokenizer=tokenizers[i],
                                                model=models[i], annoy_index=annoy_indexes[i], vocab_names=vocab_names[i], K=k)
                nnsims2 = get_nearest_neighbors(word=word, year=year, embeddings=embeddings[j], tokenizer=tokenizers[j],
                                                model=models[j], annoy_index=annoy_indexes[j], vocab_names=vocab_names[j], K=k)

                # nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
                # nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity
                nn1 = nnsims1
                nn2 = nnsims2

                inter = set.intersection(*map(set, [nn1, nn2]))

                ranks1, ranks2 = {}, {}  # for storing indices
                not_found1, not_found2 = [], []  # for storing words that are found in one list, but not in the other

                # loop over neighbors of w in embedding space 1, check if they're in the neighbors of w in embedding space 2
                # calculate their rank, if yes.
                for wp in nn1:
                    if wp in nn2:
                        ranks1[wp] = nn2.index(wp)  # index of wp in nn2
                    else:
                        # if not present, it has no index
                        not_found1.append(wp)

                # loop over neighbors of w in embedding space 2, check if they're in the neighbors of w in embedding space 1
                # calculate their rank, if yes.
                for wp in nn2:
                    if wp in nn1:
                        ranks2[wp] = nn1.index(wp)  # index of wp in nn1
                    else:
                        # if not present, it has no index
                        not_found2.append(wp)

                sum_ranks1, sum_ranks2 = 0.0, 0.0
                for wp in ranks1:
                    sum_ranks1 += ranks1[wp]
                for wp in ranks2:
                    sum_ranks2 += ranks2[wp]

                Count_neig12 = (len(nn1) * len(inter)) - sum_ranks1
                Count_neig21 = (len(nn2) * len(inter)) - sum_ranks2

                # if there are some words that are found in one list but not in the other
                # then calculate their stability using linear mapping approach
                if not_found1 != [] or not_found2 != []:
                    R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices,
                                                            model1_name=models_names[0],
                                                            model2_name=models_names[1])

                    sim_lin01, sim_lin10 = 0.0, 0.0
                    for wp in not_found1:
                        # w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                        # wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                        w_v = get_time_specific_word_embedding(word=wp, year=year, embeddings=embeddings[j],
                                                               tokenizer=tokenizers[j], model=models[j])
                        wp_v = get_time_specific_word_embedding(word=wp, year=year, embeddings=embeddings[i],
                                                                tokenizer=tokenizers[i], model=models[i])

                        w_v = w_v.flatten()
                        w_v.reshape(768, )

                        wp_v = wp_v.flatten()
                        wp_v.reshape(768, )

                        try:
                            R = R.cpu().numpy()
                        except:
                            pass

                        try:
                            val = get_cosine_sim(R.dot(wp_v), w_v)
                        except:
                            val = get_cosine_sim(R.dot(wp_v.cpu().numpy()), w_v)
                        sim_lin01 += val
                        # print('{} - {} - {}'.format(w, wp, val))

                    sim_lin01 /= len(not_found1)

                    for wp in not_found2:
                        # w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                        # wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                        w_v = get_time_specific_word_embedding(word=wp, year=year, embeddings=embeddings[i],
                                                               tokenizer=tokenizers[i], model=models[i])
                        wp_v = get_time_specific_word_embedding(word=wp, year=year, embeddings=embeddings[j],
                                                                tokenizer=tokenizers[j], model=models[j])

                        w_v = w_v.flatten()
                        w_v.reshape(768, )

                        wp_v = wp_v.flatten()
                        wp_v.reshape(768, )

                        try:
                            R_inv = R_inv.cpu().numpy()
                        except:
                            pass

                        try:
                            val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                        except:
                            val = get_cosine_sim(R_inv.dot(wp_v.cpu().numpy()), w_v)
                        sim_lin10 += val
                        # print('{} - {} - {}'.format(w, wp, val))
                    sim_lin10 /= len(not_found2)

                st_neig = (Count_neig12 + Count_neig21) / (2 * sum(
                    [i for i in range(1, k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
                st_lin = None
                if not_found1 != [] or not_found2 != []:
                    st_lin = np.mean([sim_lin01, sim_lin10])

                # calculate value of lambda
                if nn1 == nn2:
                    # when the nearest neighbours of w are exactly the same,
                    # and have the same order in embedding 1 and embedding 2
                    # lmbda = 1.0
                    st = st_neig
                    print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], word,
                                                                               st_neig, '-', st))
                elif Count_neig12 == 0 and Count_neig12 == 0:
                    # when the nearest neighbours in embedding 1 are completely
                    # not found in embedding 2, and vice versa would be also true
                    # lmbda = 0
                    st = st_lin
                    print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], word, '-',
                                                                               st_lin, st))
                else:
                    # some neighbours of w in embedding 1 are found in embedding 2,
                    # and vice versa would be true
                    lmbda = 0.5
                    st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                    print(
                        '{}-{}: {}: st_nei: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], word, st_neig,
                                                                           st_lin, st))

                similarities.append(st)

    st = np.mean(similarities)
    print('final combined stability of: {} = {}'.format(word, st))
    return st


def get_word_embedding_static(word, tokenizer, model):
    tokenized = tokenizer(word, return_tensors="pt", add_special_tokens=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_vocab_embeddings(year, vocab_path):
    """
    year: the year which will be used to get time-specific embeddings
    vocab_path: the path to the dictionary mapping each word in the vocabulary (of a fine tuned model) to its embedding
    """
    big_dict = {}
    for file in os.listdir(vocab_path):
        with open(os.path.join(vocab_path, file), 'rb') as f:
            loaded_dict = pickle.load(f)
            for k, v in loaded_dict.items():
                if year not in k:
                    continue
                if "##" in k:
                    continue
                else:
                    big_dict[k] = v

    vocab_names = [k for k in big_dict.keys()]
    vocab_embeds = big_dict.values()

    # Prepare embeddings
    embeddings_mod = []
    for i, emb in enumerate(vocab_embeds):
        if isinstance(emb, np.ndarray):  # Ensure it's a NumPy array
            pass
        else:
            emb = np.array(emb.cpu())  # Convert to NumPy if needed
        if emb.shape != (1, 768):  # Reshape to (1, 768) if necessary
            emb = emb.reshape(1, -1)
        embeddings_mod.append(emb)

    # Convert to NumPy array
    embeddings_mod = np.array(embeddings_mod)  # Shape will be (N, 1, 768)
    print("Embeddings shape:", embeddings_mod.shape)

    # Build the Annoy index
    f = embeddings_mod.shape[2]  # Dimensionality of embeddings (768)
    annoy_index = AnnoyIndex(f, 'angular')
    for i, emb in enumerate(embeddings_mod):
        try:
            annoy_index.add_item(i, emb.flatten())  # Flatten to 1D
            print("Added embedding:", i)
        except Exception as e:
            print("Error adding embedding:", e)

    annoy_index.build(20)  # Build the index with 10 trees

    return annoy_index, vocab_names


def get_nearest_neighbors(word, year, embeddings, tokenizer, model, annoy_index, vocab_names, K):

    # Query for nearest neighbors
    query = get_time_specific_word_embedding(
        word=word,
        year=year,
        embeddings=embeddings,
        tokenizer=tokenizer,
        model=model
    )
    query = query.flatten()  # Ensure query is 1D
    indices = annoy_index.get_nns_by_vector(query, K)

    nearest_neighbors = []
    # Output the nearest neighbors
    print(f"Top {K} nearest neighbors to '{word}':")
    for idx in indices:
        print(vocab_names[idx])
        nearest_neighbors.append(vocab_names[idx])

    return nearest_neighbors


def get_week_coverage_sorted(archive):
    """
    loops over the .txt files and returns a sorted list of months-weeknbs
    """
    months_weeks = set()
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"
        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        given_date = date(int(year), int(month), int(day))
        week_number = given_date.isocalendar()[1]

        month_week = f"{month}_{str(week_number)}"

        months_weeks.add(month_week)

    months_weeks_l = list(months_weeks)
    months_weeks_ls = sorted(months_weeks_l, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    print(f"months_weeks sorted for {archive}: {months_weeks_ls}")

    return months_weeks_ls


if __name__ == '__main__':
    print('hello')
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="UBC-NLP-MARBERTv2", help="model name for archives")
    parser.add_argument("--split_by", type=str, default="monthly", help="either `monthly` or `weekly`")
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word - for neighbours and combined approach")
    parser.add_argument("--method", default="combined", help="method to calculate stability - either combined/neighbors/linear")
    args = parser.parse_args()

    split_by = args.split_by
    model_name = args.model_name

    # the time-specific embedidngs
    path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/{}/'.format(model_name, split_by)
    path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/{}/'.format(model_name, split_by)

    # the trained model
    path_to_model_nahar = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/An-Nahar/{}/".format(model_name)
    path_to_model_assafir = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/As-Safir/{}/".format(model_name)

    # tokenizers for each archive
    tokenizer_nahar = AutoTokenizer.from_pretrained(path_to_model_nahar)
    tokenizer_assafir = AutoTokenizer.from_pretrained(path_to_model_assafir)

    # models for each archive
    model_nahar = AutoModelForMaskedLM.from_pretrained(path_to_model_nahar, output_hidden_states=True).cuda()
    model_nahar.eval()

    model_assafir = AutoModelForMaskedLM.from_pretrained(path_to_model_assafir, output_hidden_states=True).cuda()
    model_assafir.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(path_nahar, 'embeddings.pickle'), 'rb') as handle:
        embeddings_nahar = pickle.load(handle)
        print(len(embeddings_nahar))

    with open(os.path.join(path_assafir, 'embeddings.pickle'), 'rb') as handle:
        embeddings_assafir = pickle.load(handle)
        print(len(embeddings_assafir))

    k = int(args.k)
    save_dir = args.save_dir

    if split_by == 'weekly':
        years_nahar = get_week_coverage_sorted(archive="An-Nahar")
        years_assafir = get_week_coverage_sorted(archive="As-Safir")
        combined_years = set(years_nahar).intersection(set(years_assafir))
    else:
        combined_years = ['06', '07', '08', '09', '10', '11', '12']

    for year in tqdm(combined_years):

        model1_name = f'1982-nahar-{year}'
        model2_name = f'1982-assafir-{year}'
        model_names = [model1_name, model2_name]

        stopwords_list = []
        count = 0
        with open('top10percent.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(3016):
                if "X" not in lines[i]:
                    print(lines[i].replace("\n", ""))
                    stopwords_list.append(lines[i].replace("\n", ""))
                    count += 1
        print(count)
        stopwords_list = list(set(stopwords_list))
        print(len(stopwords_list))

        method = args.method

        save_dir_matrices = "transformation_matrices/{}-{}/".format(model_name, split_by)
        mkdir(save_dir_matrices)

        learn_stability_matrices(model1=model_nahar,
                                     model2=model_assafir,
                                     embeddings_1=embeddings_nahar,
                                     embeddings_2=embeddings_assafir,
                                     tokenizer_1=tokenizer_nahar,
                                     tokenizer_2=tokenizer_assafir,
                                     model1_name=model1_name,
                                     model2_name=model2_name,
                                     subwords=stopwords_list,
                                     save_dir=save_dir_matrices)
