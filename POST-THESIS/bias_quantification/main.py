import torch
import matplotlib.pyplot as plt
import os
from helper import *
import pickle
import arabic_reshaper
from bidi.algorithm import get_display
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from normalization import ArabicNormalizer



def mkdir(direc):
    if not os.path.exists(direc):
        os.makedirs(direc, exist_ok=True)


def plot_differences(list1, list2, model, features, xlabel, ylabel, save_dir, fig_name):

    v1 = np.mean([model[word] for word in list1], axis=0)
    v2 = np.mean([model[word] for word in list2], axis=0)
    x = []
    y = []
    for word in features:
        x.append(calc_distance_between_vectors(model[word], v1))
        y.append(calc_distance_between_vectors(model[word], v2))
    C = [x_ - y_ for x_, y_ in zip(x, y)]
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.xlim(min(y) - 0.05, max(y) + 0.05)
    # top_idx = random.sample(list(np.argsort(C)[-50:]), 20)
    # bottom_idx = random.sample(list(np.argsort(C)[:50]), 20)
    for i, label in enumerate(features):
        # if i in top_idx or i in bottom_idx:
        ax.annotate(get_display(arabic_reshaper.reshape(label)), (x[i], y[i]))

    ax.plot([min(y) -0.05, max(x) +0.15], [min(y) -0.05, max(x) + 0.05], ls="--", c=".3")

    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
    plt.close()
    # return C, top_idx


def plot_embedding_bias_over_time(biases, ylabel, event_name, distance_type, save_dir):
    for model_name in biases:
        for archive_name in biases[model_name]:
            print(biases[model_name][archive_name])
            plt.plot(list(range(len(biases[model_name][archive_name]))), biases[model_name][archive_name], marker='o', label=f"EB for {archive_name}")

        plt.hlines(y=0, xmin=0, xmax=len(biases[model_name][archive_name]), colors='black', linestyles='--', lw=2, label='EB=0')

        plt.legend()

        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        plt.grid(axis='y')

        plt.xticks(list(range(len(biases[model_name][archive_name]))), ['1982-{}'.format(m) for m in ['06', '07', '08', '09', '10', '11', '12']], rotation=45)
        plt.ylabel(ylabel)

        mkdir(save_dir)
        fig_name = '{}_{}_{}'.format(event_name, model_name, distance_type)
        plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
        plt.close()


# def get_word_embedding(word, tokenizer):
#     # Tokenize the word
#     inputs = tokenizer(word, return_tensors="pt")
#     input_ids = inputs["input_ids"].cuda()
#
#     # Get hidden states
#     with torch.no_grad():
#         outputs = model(input_ids)
#         hidden_states = outputs.hidden_states  # All layers' hidden states
#
#     # Use the last 4 layers and take the mean for the word embedding
#     word_embedding = torch.stack(hidden_states[-4:]).mean(0)[0, 1:-1].mean(0)  # Ignore [CLS] and [SEP] tokens
#     return word_embedding.cpu().numpy()

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


def embedding_bias(embeddings, target_list1, target_list2, neutral_list, distance_type, tokenizer, model, device="cpu"):
    means = []
    years = ['06', '07', '08', '09', '10', '11', '12']
    for year in years:
        v1, v2 = [], []
        for word in target_list1:
            embedding = get_time_specific_word_embedding(word=word, year=year, embeddings=embeddings, tokenizer=tokenizer, model=model)
            # Check and adjust shape
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding).to(device)
            if embedding.shape != (1, 768):
                embedding = embedding.unsqueeze(0).to(device)
            v1.append(embedding.to(device))

        for word in target_list2:
            embedding = get_time_specific_word_embedding(word=word, year=year, embeddings=embeddings, tokenizer=tokenizer, model=model)
            # Check and adjust shape
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding).to(device)
            if embedding.shape != (1, 768):
                embedding = embedding.unsqueeze(0).to(device)
            v2.append(embedding.to(device))

        # Ensure all tensors are concatenated on the same device
        v1 = torch.cat(v1, dim=0)  # [num_words, 768]
        v2 = torch.cat(v2, dim=0)  # [num_words, 768]

        # Compute mean embeddings
        v1 = v1.mean(dim=0).cpu().numpy()
        v2 = v2.mean(dim=0).cpu().numpy()

        x = []
        y = []
        for word in neutral_list:
            try:
                word_emb = get_time_specific_word_embedding(word=word, year=year, embeddings=embeddings, tokenizer=tokenizer, model=model)
                # Check and adjust shape
                if isinstance(word_emb, np.ndarray):
                    word_emb = torch.tensor(word_emb).to(device)
                if word_emb.shape != (1, 768):
                    word_emb = word_emb.unsqueeze(0)
                x.append(calc_distance_between_vectors(v1, word_emb.squeeze(), distype=distance_type))
                y.append(calc_distance_between_vectors(v2, word_emb.squeeze(), distype=distance_type))

            except Exception as e:
                print(f"Error processing word {word}: {e}")
                pass

        C = [x_ - y_ for x_, y_ in zip(x, y)]
        print(C)
        means.append(np.mean(C))

    return means



if __name__ == '__main__':
    entities_target_neutral_lists = {
        'Arab World Slacktivism': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities1.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_1.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_1.txt',
            ]
        },
        'Bashir Gemayel Election': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities2.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_2.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_2.txt',
            ]
        },
        'Israeli Invasion': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities3.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_3.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_3.txt',
            ]
        },
        'Palestinians': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities4.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_4.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_4.txt',
            ]
        },
        'Philip Habib': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities5.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_5.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_5.txt',
            ]
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UBC-NLP-MARBERTv2', help='name of the model for which embeddings are stored')
    args = parser.parse_args()

    models = [
        "UBC-NLP-MARBERTv2",
        "qarib-bert-base-qarib",
        "UBC-NLP-MARBERT",
        "UBC-NLP-ARBERT",
        "aubmindlab-bert-base-arabert",
        "aubmindlab-bert-base-arabertv02-twitter",
        "aubmindlab-bert-base-arabertv2",
        "aubmindlab-bert-base-arabertv01",
        "aubmindlab-bert-base-arabertv02",
    ]

    for event_name in entities_target_neutral_lists:
        biases = {}
        for dist_type in ['norm', 'cossim']:
            for model_name in models:
                biases[model_name] = {}

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

                print(f"Processing {event_name}")
                file_entities = entities_target_neutral_lists[event_name]['neutral_list']
                file_negconn = entities_target_neutral_lists[event_name]['target_list'][0]
                file_posconn = entities_target_neutral_lists[event_name]['target_list'][1]

                with open(file_entities, encoding='utf-8') as f:
                    entities = f.readlines()
                    entities = [e.strip().replace("\n", "") for e in entities]

                with open(file_negconn, encoding='utf-8') as f:
                    negconn = f.readlines()
                    negconn = [e.strip().replace("\n", "") for e in negconn]

                with open(file_posconn, encoding='utf-8') as f:
                    posconn = f.readlines()
                    posconn = [e.strip().replace("\n", "") for e in posconn]


                biases_nahar = embedding_bias(embeddings_nahar, target_list1=negconn, target_list2=posconn, neutral_list=entities, distance_type=dist_type,
                                              tokenizer=tokenizer_nahar, model=model_nahar)
                biases[model_name]['An-Nahar'] = biases_nahar

                biases_assafir = embedding_bias(embeddings_assafir, target_list1=negconn, target_list2=posconn, neutral_list=entities, distance_type=dist_type,
                                                tokenizer=tokenizer_assafir, model=model_assafir)
                biases[model_name]['As-Safir'] = biases_assafir
                # read each word
                # for event_name in entities_target_neutral_lists:

            plot_embedding_bias_over_time(biases=biases,
                                          ylabel=f"Avg. Embedding Bias for {event_name}",
                                          event_name=event_name,
                                          distance_type=dist_type,
                                          save_dir="plots/"
                                          )
