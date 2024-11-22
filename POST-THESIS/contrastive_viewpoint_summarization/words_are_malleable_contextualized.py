import torch
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import argparse
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
from normalization import ArabicNormalizer
from annoy import AnnoyIndex


def get_word_embedding_static(word, tokenizer, model):
    """ Get static word embedding for a single word """
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
    return dot(v1, v2) / (norm(v1) * norm(v2))


def load_linear_mapping_matrices(dir_name, mat_name, model1_name, model2_name):
    """
    load the linear transformation matrices between one embedding and another
    :param dir_name: name of the directory where the matrices are stored
    :param mat_name: file name used to store the matrices (this is a suffix; the
                    inverse transformation matrix has an extra substring '_inv')
    :return the transformation matrix R and its inverse R_inv
    """
    R = np.load(os.path.join(dir_name, mat_name + '_{}_{}.npy').format(model1_name, model2_name))
    R_inv = np.load(os.path.join(dir_name, mat_name + '_{}_{}_inv.npy'.format(model1_name, model2_name)))
    return R, R_inv


def get_stability_linear_mapping_one_word(model1, model2, model1_name, model2_name, mat_name, dir_name_matrices, w):
    ''' gets the stability values of one oword  only
        assumes that transformation matrices are already computed
    '''
    R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=model1_name, model2_name=model2_name)

    w0 = model1.get_word_vector(w) if ' ' not in w else model1.get_sentence_vector(w)
    w1 = model2.get_word_vector(w) if ' ' not in w else model2.get_sentence_vector(w)

    # the stability of a word is basically the stability of the vector to its mapped
    # vector after applying the mapping back and forth
    sim01 = get_cosine_sim(R_inv.dot(R.dot(w0)), w0)
    sim10 = get_cosine_sim(R.dot(R_inv.dot(w1)), w1)

    stability = (sim01 + sim10) / 2

    print('stability (linear) of the word {}: {}'.format(w, stability))

    return sim01, sim10


def learn_stability_matrices(model1, model2, embeddings_1, embeddings_2, tokenizer_1, tokenizer_2, model1_name, model2_name, subwords, num_steps, mat_name):
    """
    Get the stability by applying a transformation matrix that maps word w in embedding space 1
    to word w in embedding space 2. Transformation matrix is learned through gradient descent optimization
    by applying Forbenious norm in order to minimize loss. It uses a set of training words (stopwords or
    any set of frequent words that have a stable meaning regardless of time/source)

    :param model1: trained model - embedding 1
    :param model2: trained model - embedding 2
    :param subwords: list of words/stopwords to consider for learning the transformation matrix
    :param num_steps: number of training steps for gradient descent optimization
    :param w: the word we want to get the stability for
    :return:
    """
    # print('Getting linear stability for the word {}'.format(w))
    print('Calculating stability values (linear)')

    def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
        '''
        Inputs:
            X: a matrix of dimension (m,n) where the columns are the English embeddings.
            Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
            train_steps: positive int - describes how many steps will gradient descent algorithm do.
            learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
        Outputs:
            R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
        '''
        # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
        # R is a square matrix with length equal to the number of dimensions in th  word embedding
        R = np.random.rand(X.shape[1], X.shape[1])

        losses = []
        t1 = time.time()
        for i in range(train_steps):
            loss = compute_loss(X, Y, R)

            if i % 25 == 0:
                print(f"loss at iteration {i} is: {loss:.4f}")

            losses.append(loss)
            # use the function that you defined to compute the gradient
            gradient = compute_gradient(X, Y, R)

            # update R by subtracting the learning rate times gradient
            R -= learning_rate * gradient
        t2 = time.time()
        print('time taken to learn the transformation matrix: {} mins'.format((t2-t1)/60))
        return R, losses

    def compute_gradient(X, Y, R):
        '''
        Inputs:
           X: a matrix of dimension (m,n) where the columns are the English embeddings.
           Y: a matrix of dimension (m,n) where the columns correspond to the French embeddings.
           R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
           g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
        '''
        # m is the number of rows in X
        rows, columns = X.shape

        # gradient is X^T(XR - Y) * 2/m
        gradient = (np.dot(X.T, np.dot(X, R) - Y) * 2) / rows
        assert gradient.shape == (columns, columns)

        return gradient

    def compute_loss(X, Y, R):
        '''
        Inputs:
           X: a matrix of dimension (m,n) where the columns are the English embeddings.
           Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
           R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
        Outputs:
           L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
        '''
        # m is the number of rows in X
        m = len(X)

        # diff is XR - Y
        diff = np.dot(X, R) - Y

        # diff_squared is the element-wise square of the difference
        diff_squared = diff ** 2

        # sum_diff_squared is the sum of the squared elements
        sum_diff_squared = diff_squared.sum()

        # loss is the sum_diff_squared divided by the number of examples (m)
        loss = sum_diff_squared / m
        return loss

    def get_transformation_matrices():
        # create the matrices X and Y of source embeddings i and target embeddings j
        X, Y = [], []

        for w in subwords:

            # x = model1.get_word_vector(w) if ' ' not in w else model1.get_sentence_vector(w)
            # y = model2.get_word_vector(w) if ' ' not in w else model2.get_sentence_vector(w)

            x = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_1, tokenizer=tokenizer_1, model=model1)
            y = get_time_specific_word_embedding(word=w, year=year, embeddings=embeddings_2, tokenizer=tokenizer_2, model=model2)

            X.append(x)
            Y.append(y)

        X = np.vstack([x.cpu() for x in X])
        Y = np.vstack([y.cpu() for y in Y])

        # get the transformation matrix R
        R, losses = align_embeddings(X=X, Y=Y, train_steps=num_steps)
        R_inv = np.linalg.inv(R)

        return R, R_inv, losses

    def save_matrices(R, R_inv):
        # save transformation matrix and its inverse
        np.save(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name)), R)
        np.save(os.path.join(dir_name_matrices, mat_name + '_{}_{}_inv.npy'.format(model1_name, model2_name)), R_inv)

    def plot_losses(losses):
        # dir_name_losses = os.path.join(save_dir, 'loss_plots/')
        # mkdir(losses)  # create directory of does not already exist
        plt.plot(range(len(losses)), losses)
        plt.xlabel('steps')
        plt.ylabel('Loss')

        plt.savefig(os.path.join(dir_name_losses, mat_name + '_{}_{}.png'.format(model1_name, model2_name)))
        plt.close()

    # if the matrices exist, return since they will be loaded in the combined method
    if os.path.exists(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name))):
        print('matrices found in {}'.format(os.path.join(dir_name_matrices, mat_name + '_{}_{}.npy'.format(model1_name, model2_name))))
        return

    # get the transformation matrix and its inverse
    R, R_inv, losses = get_transformation_matrices()
    print('got transformation matrices ...')

    plot_losses(losses)
    print('saved plot of loss ...')

    # save the transformation matrices
    save_matrices(R=R, R_inv=R_inv)
    print('saved transformation matrices ...')
    return


def get_stability_linear_mapping(model1, model2, model1_name, model2_name, mat_name, words_path=None, save_dir='results/', file_name='stabilities_linear'):
    print('Calculating stability values (linear)')

    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))

    else:
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
        s_lin12, s_lin21 = get_stability_linear_mapping_one_word(model1, model2, model1_name=model1_name, model2_name=model2_name,
                                                                 mat_name=mat_name, dir_name_matrices=dir_name_matrices, w=w)

        stability = (s_lin12 + s_lin21) / 2
        stabilities[w] = stability

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_combined_one_word(w, models, models_names, mat_name, dir_name_matrices, k=50):
    """ gets the combined stability value of a certain word w """
    if len(models) > 2:
        similarities = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    nnsims1 = models[i].get_nearest_neighbors(w, k)
                    nnsims2 = models[j].get_nearest_neighbors(w, k)

                    nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
                    nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

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
                        R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[i], model2_name=models_names[j])

                        sim_lin01, sim_lin10 = 0.0, 0.0
                        for wp in not_found1:
                            w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                            wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                            val = get_cosine_sim(R.dot(wp_v), w_v)
                            sim_lin01 += val
                            print('{} - {} - {}'.format(w, wp, val))
                        sim_lin01 /= len(not_found1)

                        for wp in not_found2:
                            w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                            wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                            val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                            sim_lin10 += val
                            print('{} - {} - {}'.format(w, wp, val))
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
                        print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
                    elif Count_neig12 == 0 and Count_neig12 == 0:
                        # when the nearest neighbours in embedding 1 are completely
                        # not found in embedding 2, and vice versa would be also true
                        # lmbda = 0
                        st = st_lin
                        print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
                    else:
                        # some neighbours of w in embedding 1 are found in embedding 2,
                        # and vice versa would be true
                        lmbda = 0.5
                        st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                        print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

                    similarities.append(st)

        return np.mean(similarities)

    else:
        nnsims1 = models[0].get_nearest_neighbors(w, k)
        nnsims2 = models[1].get_nearest_neighbors(w, k)

        nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
        nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

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
            R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name,
                                                    model1_name=models_names[0], model2_name=models_names[1])

            sim_lin01, sim_lin10 = 0.0, 0.0
            for wp in not_found1:
                w_v = models[1].get_word_vector(w) if ' ' not in w else models[1].get_sentence_vector(w)
                wp_v = models[0].get_word_vector(wp) if ' ' not in wp else models[0].get_sentence_vector(wp)

                val = get_cosine_sim(R.dot(wp_v), w_v)
                sim_lin01 += val
                # print('{} - {} - {}'.format(w, wp, val))
            sim_lin01 /= len(not_found1)

            for wp in not_found2:
                w_v = models[0].get_word_vector(w) if ' ' not in w else models[0].get_sentence_vector(w)
                wp_v = models[1].get_word_vector(wp) if ' ' not in wp else models[1].get_sentence_vector(wp)

                val = get_cosine_sim(R_inv.dot(wp_v), w_v)
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
            print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
        elif Count_neig12 == 0 and Count_neig12 == 0:
            # when the nearest neighbours in embedding 1 are completely
            # not found in embedding 2, and vice versa would be also true
            # lmbda = 0
            st = st_lin
            print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
        else:
            # some neighbours of w in embedding 1 are found in embedding 2,
            # and vice versa would be true
            lmbda = 0.5
            st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
            print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

        return st


def get_stability_combined(models, models_names, mat_name, words_path=None, k=50, save_dir='results/',
                           file_name='stabilities_combined'):
    """
    Method that applies Algorithm 2 (Combined method) of Words are Malleable paper by Azarbonyad et al (2017)
    However, we have modified the code from the one presented in the paper in few places
    because of the fact that we can get vectors of words that are OOV (out-of-vocabulary)
    therefore, no need for getting the words in the intersection of two embedding spaces as
    most of the words we were interested in are in fact OOV due to spelling errors
    caused by applying OCR to digitized historical documents.

    Also, the code is formatted to work with t=1, i.e., we are only using direct neighbors of words (t=1)
    and not neighbors of neighbors (t >= 1). Therefore, we removed MinMax normalization step

    :param model1: fasttext model representing embedding space 1
    :param model2: fasttext model representing embedding space 2
    :param mat_name: name of the transformation matrix calculated via linear mapping that maps from embedding space 1 to embedding space 2
    :param words_path: path to file that contains words that are OOV
    :param k: number of nearest neighbors to extract per word
    :param t: number of iterations
    :param save_dir: directory to save final stabilities per word (saved as a python dictionary)
    :param file_name: name of the final stabilities file (saved as a python dictionary)
    :return:
    """
    print('Calculating stability values (combined) - k={}'.format(k))
    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))

    else:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        words = [w[:-1] for w in words if '\n' in w]  # remove '\n'
        words = [w for w in words if w.strip() != '']
        vocab = [w.strip() for w in words]

    print('len of vocab: {}'.format(len(vocab)))
    stabilities = {}
    for w in vocab:
        if len(models) > 2:
            similarities = []
            for i in range(len(models)):
                for j in range(len(models)):
                    if i != j:
                        nnsims1 = models[i].get_nearest_neighbors(w, k)
                        nnsims2 = models[j].get_nearest_neighbors(w, k)

                        nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
                        nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

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
                            R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[i], model2_name=models_names[j])

                            sim_lin01, sim_lin10 = 0.0, 0.0
                            for wp in not_found1:
                                w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                                wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                                val = get_cosine_sim(R.dot(wp_v), w_v)
                                sim_lin01 += val
                                # print('{} - {} - {}'.format(w, wp, val))
                            sim_lin01 /= len(not_found1)

                            for wp in not_found2:
                                w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                                wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                                val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                                sim_lin10 += val
                                # print('{} - {} - {}'.format(w, wp, val))
                            sim_lin10 /= len(not_found2)

                        st_neig = (Count_neig12 + Count_neig21) / (2 * sum([i for i in range(1,
                                                                                             k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
                        st_lin = None
                        if not_found1 != [] or not_found2 != []:
                            st_lin = np.mean([sim_lin01, sim_lin10])

                        # calculate value of lambda
                        if nn1 == nn2:
                            # when the nearest neighbours of w are exactly the same,
                            # and have the same order in embedding 1 and embedding 2
                            # lmbda = 1.0
                            st = st_neig
                            print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, st_neig, '-', st))
                        elif Count_neig12 == 0 and Count_neig12 == 0:
                            # when the nearest neighbours in embedding 1 are completely
                            # not found in embedding 2, and vice versa would be also true
                            # lmbda = 0
                            st = st_lin
                            print('{}-{}: {}: st_neigh: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, '-', st_lin, st))
                        else:
                            # some neighbours of w in embedding 1 are found in embedding 2,
                            # and vice versa would be true
                            lmbda = 0.5
                            st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                            print('{}-{}: {}: st_nei: {}, st_lin: {}, st: {}'.format(models_names[i], models_names[j], w, st_neig, st_lin, st))

                        similarities.append(st)

            stabilities[w] = np.mean(similarities)
            print('final combined stability of: {} = {}'.format(w, np.mean(similarities)))

        else:
            nnsims1 = models[0].get_nearest_neighbors(w, k)
            nnsims2 = models[1].get_nearest_neighbors(w, k)

            nn1 = [n[1] for n in nnsims1]  # get only the neighbour, not the similarity
            nn2 = [n[1] for n in nnsims2]  # get only the neighbour, not the similarity

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
                R, R_inv = load_linear_mapping_matrices(dir_name=dir_name_matrices, mat_name=mat_name, model1_name=models_names[0], model2_name=models_names[1])

                sim_lin01, sim_lin10 = 0.0, 0.0
                for wp in not_found1:
                    w_v = models[1].get_word_vector(w) if ' ' not in w else models[1].get_sentence_vector(w)
                    wp_v = models[0].get_word_vector(wp) if ' ' not in wp else models[0].get_sentence_vector(wp)

                    val = get_cosine_sim(R.dot(wp_v), w_v)
                    sim_lin01 += val
                    # print('{} - {} - {}'.format(w, wp, val))
                sim_lin01 /= len(not_found1)

                for wp in not_found2:
                    w_v = models[0].get_word_vector(w) if ' ' not in w else models[0].get_sentence_vector(w)
                    wp_v = models[1].get_word_vector(wp) if ' ' not in wp else models[1].get_sentence_vector(wp)

                    val = get_cosine_sim(R_inv.dot(wp_v), w_v)
                    sim_lin10 += val
                    # print('{} - {} - {}'.format(w, wp, val))
                sim_lin10 /= len(not_found2)

            st_neig = (Count_neig12 + Count_neig21) / (2 * sum([i for i in range(1, k + 1)]))  # this is 2 * (k)(k+1) where k is the number of nearest neighbors
            st_lin = None
            if not_found1 != [] or not_found2 != []:
                st_lin = np.mean([sim_lin01, sim_lin10])

            # calculate value of lambda
            if nn1 == nn2:
                # when the nearest neighbours of w are exactly the same,
                # and have the same order in embedding 1 and embedding 2
                # lmbda = 1.0
                st = st_neig
                print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, st_neig, '-', st))
            elif Count_neig12 == 0 and Count_neig12 == 0:
                # when the nearest neighbours in embedding 1 are completely
                # not found in embedding 2, and vice versa would be also true
                # lmbda = 0
                st = st_lin
                print('{}: st_neigh: {}, st_lin: {}, st: {}'.format(w, '-', st_lin, st))
            else:
                # some neighbours of w in embedding 1 are found in embedding 2,
                # and vice versa would be true
                lmbda = 0.5
                st = (lmbda * st_neig) + ((1 - lmbda) * st_lin)
                print('{}: st_nei: {}, st_lin: {}, st: {}'.format(w, st_neig, st_lin, st))

            stabilities[w] = st
            print('final combined stability of: {} = {}'.format(w, st))

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


def get_stability_neighbors_one_word(w, models, k=50):
    similarities = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                nnsims1 = models[i].get_nearest_neighbors(w, k)
                nnsims2 = models[j].get_nearest_neighbors(w, k)

                nn1 = [n[1] for n in nnsims1]  # get only the neighbor, not the similarity
                nn2 = [n[1] for n in nnsims2]  # get only the neighbor, not the similarity

                sim1, sim2 = 0, 0
                for wp in nn2:
                    w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                    wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                    sim2 += get_cosine_sim(w_v, wp_v)

                sim2 /= len(nn2)

                for wp in nn1:
                    w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                    wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                    sim1 += get_cosine_sim(w_v, wp_v)

                sim1 /= len(nn1)

                # calculate stability as the average of both similarities
                similarities.append(sim1)
                similarities.append(sim2)

    # calculate the word's stability
    st = np.mean(similarities)
    print('Neighbor\'s approach stability: {}: {}'.format(w, st))


def get_stability_neighbors(models, words_path=None, k=50, save_dir='results/', file_name='stabilities_neighbors'):
    """
    Method that applies Algorithm 1 (Neighbors method) of Words are Malleable paper by Azarbonyad et al (2017)
    However, we have modified the code from the one presented in the paper in few places
    because of the fact that we can get vectors of words that are OOV (out-of-vocabulary),
    therefore, no need for getting the words in the intersection of two embedding spaces as
    most of the words we were interested in are in fact OOV due to spelling errors
    caused by applying OCR to digitized historical documents.

    Also, the code is formatted to work with t=1, i.e., we are only using direct neighbors of words (t=1)
    and not neighbors of neighbors (t >= 1). Therefore, we removed MinMax normalization step

    :param model1: fasttext model representing embedding space 1
    :param model2: fasttext model representing embedding space 1
    :param k: number of nearest neighbors to extract per word
    :param t: number of iterations
    :param save_dir: directory to save final stabilities per word (saved as a python dictionary)
    :param file_name: name of the final stabilities file (saved as a python dictionary)
    :return:
    """

    print('Calculating stability values (neighbors) - k={}'.format(k))
    if words_path is None:
        all_vocab = []
        all_vocab.append(model1.words)
        all_vocab.append(model2.words)
        vocab = list(set.intersection(*map(set, all_vocab)))
        print('len of vocab: {}'.format(len(vocab)))
    else:
        with open(words_path, 'r', encoding='utf-8') as f:
            words = f.readlines()
        words = [w[:-1] for w in words if '\n' in w] # remove '\n'
        words = [w for w in words if w.strip() != '']
        vocab = [w for w in words]
        print('len of vocab: {}'.format(len(vocab)))

    # dictionary mapping each word w to its stability value. Before any processing, stability of any word w is 1.0
    stabilities = {}

    for w in vocab:
        similarities = []
        for i in range(len(models)):
            for j in range(len(models)):
                if i != j:
                    nnsims1 = models[i].get_nearest_neighbors(w, k)
                    nnsims2 = models[j].get_nearest_neighbors(w, k)

                    nn1 = [n[1] for n in nnsims1]  # get only the neighbor, not the similarity
                    nn2 = [n[1] for n in nnsims2]  # get only the neighbor, not the similarity

                    sim1, sim2 = 0, 0
                    for wp in nn2:
                        w_v = models[i].get_word_vector(w) if ' ' not in w else models[i].get_sentence_vector(w)
                        wp_v = models[i].get_word_vector(wp) if ' ' not in wp else models[i].get_sentence_vector(wp)

                        sim2 += get_cosine_sim(w_v, wp_v)

                    sim2 /= len(nn2)

                    for wp in nn1:
                        w_v = models[j].get_word_vector(w) if ' ' not in w else models[j].get_sentence_vector(w)
                        wp_v = models[j].get_word_vector(wp) if ' ' not in wp else models[j].get_sentence_vector(wp)

                        sim1 += get_cosine_sim(w_v, wp_v)

                    sim1 /= len(nn1)

                    # calculate stability as the average of both similarities
                    similarities.append(sim1)
                    similarities.append(sim2)

        # calculate the word's stability
        st = np.mean(similarities)
        stabilities[w] = st
        print('Neighbor\'s approach stability: {}: {}'.format(w, st))

    # save the stabilities dictionary
    mkdir(save_dir)
    with open(os.path.join(save_dir, '{}.pkl'.format(file_name)), 'wb') as handle:
        pickle.dump(stabilities, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stabilities


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
    print('hello')
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="UBC-NLP-MARBERT", help="model name for archives")

    # neighbor and combined approach (words_file is needed by linear approach as well)
    parser.add_argument("--k", default=100, help="number of nearest neighbors to consider per word - for neighbours and combined approach")
    parser.add_argument("--save_dir", default="results_diachronic_new/", help="directory to save stabilities dictionary")

    # linear approach
    parser.add_argument("--num_steps", default=80000, help="number of training steps for gradient descent optimization")
    parser.add_argument("--mat_name", default="trans", help="prefix of the name of the transformation matrices to be saved - linear mapping approach")

    parser.add_argument("--month_name", default="06", help="the month to which we should aligh Nahar_month to Assafir_month")

    # name of the method to be used ('combined' vs. 'linear' vs. neighbor)
    parser.add_argument("--method", default="combined", help="method to calculate stability - either combined/neighbors/linear")
    args = parser.parse_args()

    model_name = args.model_name
    path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/'.format(model_name)
    path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/'.format(model_name)

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

    model1_name = '1982-nahar'
    model2_name = '1982-assafir'

    # Example texts
    year = "06"
    word = "الحكم"

    # path to vocabulary embeddings
    path_nahar_vocab = "vocabulary_embeddings_copy/An-Nahar/{}".format(model_name)
    path_assafir_vocab = "vocabulary_embeddings_copy/As-Safir/{}".format(model_name)

    # with open(os.path.join(path_nahar_vocab, 'An-Nahar_{}.pickle'.format(model_name)), 'rb') as handle:
    #     vocab_embeddings_nahar = pickle.load(handle)
    #
    # with open(os.path.join(path_assafir_vocab, 'As-Safir_{}.pickle'.format(model_name)), 'rb') as handle:
    #     vocab_embeddings_assafir = pickle.load(handle)

    # with open("/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/UBC-NLP-MARBERTv2/embeddings.pickle", "rb") as handle:
    #     vocab_embeddings_nahar = pickle.load(handle)

    # candidates = [k for k in vocab_embeddings_nahar if year in k]
    #
    # # Prepare embeddings
    # embeddings_mod = []
    # for c in candidates:
    #     emb = vocab_embeddings_nahar[c]
    #     if isinstance(emb, np.ndarray):  # Ensure it's a NumPy array
    #         pass
    #     else:
    #         emb = np.array(emb)  # Convert to NumPy if needed
    #     if emb.shape != (1, 768):  # Reshape to (1, 768) if necessary
    #         emb = emb.reshape(1, -1)
    #     embeddings_mod.append(emb)
    #
    # # Convert to NumPy array
    # embeddings_mod = np.array(embeddings_mod)  # Shape will be (N, 1, 768)
    # print("Embeddings shape:", embeddings_mod.shape)
    #
    # # Build the Annoy index
    # f = embeddings_mod.shape[2]  # Dimensionality of embeddings (768)
    # annoy_index = AnnoyIndex(f, 'angular')
    #
    # for i, emb in enumerate(embeddings_mod):
    #     try:
    #         annoy_index.add_item(i, emb.flatten())  # Flatten to 1D
    #         print("Added embedding:", i)
    #     except Exception as e:
    #         print("Error adding embedding:", e)
    #
    # annoy_index.build(10)  # Build the index with 10 trees
    #
    # # Query for nearest neighbors
    # query = get_time_specific_word_embedding(
    #     word="الحكم",
    #     year=year,
    #     embeddings=embeddings_nahar,
    #     tokenizer=tokenizer_nahar
    # )
    # query = query.flatten()  # Ensure query is 1D
    # K = 100
    # indices = annoy_index.get_nns_by_vector(query, K)
    #
    # # Output the nearest neighbors
    # print(f"Top {K} nearest neighbors to 'الحكم':")
    # for idx in indices:
    #     print(candidates[idx])

    stopwords_list = stopwords.words('arabic') # stopwords for linear mapping approach
    num_steps = args.num_steps # number of training steps for gradient descent
    mat_name = args.mat_name  # prefix for the matrix name of the transformation matrix for saving purposes

    save_dir_combined_neighbor = os.path.join(save_dir, '{}-{}_{}-{}/k{}/'.format(model1_name, year, model2_name, year, k))  # to save stability dictionaries of combined and neighbors approach
    save_dir_linear = os.path.join(save_dir, "{}-{}_{}-{}/linear_numsteps{}/".format(model1_name, year, model2_name, year, num_steps))  # to save transformation matrices
    save_dir_linear_stabilities = os.path.join(save_dir, '{}-{}_{}-{}/'.format(model1_name, year, model2_name, year))  # to save stability dictionary of linear mapping approach

    # sub-directories for saving the transformation matrices
    dir_name_matrices = os.path.join(save_dir_linear, 'matrices/')
    dir_name_losses = os.path.join(save_dir_linear, 'loss_plots/')

    mkdir(save_dir_linear)             # create directory if does not already exist
    mkdir(save_dir_combined_neighbor)  # create directory if does not already exist
    mkdir(dir_name_matrices)           # create directory if does not already exist
    mkdir(dir_name_losses)             # create directory if does not already exist

    method = args.method

    if method == 'linear' or method == 'combined':

        learn_stability_matrices(model1=model_nahar,
                                 model2=model_assafir,
                                 embeddings_1=embeddings_nahar,
                                 embeddings_2=embeddings_assafir,
                                 tokenizer_1=tokenizer_nahar,
                                 tokenizer_2=tokenizer_assafir,
                                 model1_name=model1_name,
                                 model2_name=model2_name,
                                 subwords=stopwords_list,
                                 num_steps=num_steps,
                                 mat_name=mat_name)

    #     get_stability_linear_mapping(model1=model_nahar,
    #                                  model2=model_assafir,
    #                                  model1_name=model1_name,
    #                                  model2_name=model2_name, mat_name=mat_name,
    #                                  words_path=keywords_path, save_dir=save_dir_linear_stabilities,
    #                                  file_name='stabilities_linear')
    #
    # if method == "combined":
    #     # run the combined algorithm
    #     get_stability_combined(models=models, models_names=models_names, mat_name=mat_name, words_path=keywords_path,
    #                            k=k, save_dir=save_dir_combined_neighbor, file_name='stabilities_combined')
    #
    # if method == "neighbor":
    #     get_stability_neighbors(models=models, words_path=keywords_path,
    #                             k=k, save_dir=save_dir_combined_neighbor, file_name='stabilities_neighbor')
