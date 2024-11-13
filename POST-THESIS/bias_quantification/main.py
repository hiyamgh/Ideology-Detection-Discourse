import torch
import matplotlib.pyplot as plt
import os
from helper import *
import pickle
import arabic_reshaper
from bidi.algorithm import get_display
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM


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


# def plot_embedding_bias_over_time(biases_nahar, biases_assafir, ylabel, save_dir, fig_name):
#     biases_nahar.extend([np.nan, np.nan])
#     plt.plot(list(range(len(biases_assafir))), biases_assafir, marker='o', label="assafir")
#     plt.plot(list(range(len(biases_nahar))), biases_nahar, marker='o', label="nahar")
#     plt.hlines(y=0, xmin=0, xmax=len(biases_assafir), colors='black', linestyles='--', lw=2, label='EB=0')
#
#     plt.legend()
#
#     fig = plt.gcf()
#     fig.set_size_inches(12, 6)
#     plt.grid(axis='y')
#
#     plt.xticks(list(range(len(biases_assafir))), list(range(1982, 2012)), rotation='vertical')
#     plt.ylabel(ylabel)
#
#     mkdir(save_dir)
#     plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
#     plt.close()

def plot_embedding_bias_over_time(biases, ylabel, save_dir, fig_name):
    plt.plot(list(range(len(biases))), biases, marker='o', label="Avg. Embedding bias")
    plt.hlines(y=0, xmin=0, xmax=len(biases), colors='black', linestyles='--', lw=2, label='EB=0')

    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.grid(axis='y')

    plt.xticks(list(range(len(biases))), ['1982-{}'.format(m) for m in ['06', '07', '08', '09', '10', '11', '12']], rotation='vertical')
    plt.ylabel(ylabel)

    mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
    plt.close()


# def embedding_bias(list1, list2, features, models, archive_name):
#     means = []
#     # bounds = []
#     # values = []
#     # year = []
#     # for index, model in enumerate(models):
#     for year in models[archive_name]:
#         v1 = np.mean([models[archive_name][year][word] for word in list1 if word in models[archive_name][year]], axis=0)
#         #v = calculate_vectors(model, male_words, female_words)
#         v2 = np.mean([models[archive_name][year][word] for word in list2 if word in models[archive_name][year]], axis=0)
#         x = []
#         y = []
#         for word in features:
#             try:
#                 x.append(calc_distance_between_vectors(v1, models[archive_name][year][word]))
#                 y.append(calc_distance_between_vectors(v2, models[archive_name][year][word]))
#             except:
#                 pass
#         C = [x_ - y_ for x_, y_ in zip(x, y)]
#         # values.append(C)
#         means.append(np.mean(C))
#         # bounds.append(pb.bootstrap(C, confidence=0.95, iterations=1000, sample_size=.9, statistic=np.mean))
#     # return values, means, bounds
#     return means

def get_word_embedding(word, tokenizer):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    # Get hidden states
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states  # All layers' hidden states

    # Use the last 4 layers and take the mean for the word embedding
    word_embedding = torch.stack(hidden_states[-4:]).mean(0)[0, 1:-1].mean(0)  # Ignore [CLS] and [SEP] tokens
    return word_embedding.cpu().numpy()


def embedding_bias(embeddings_nahar, embeddings_assafir, target_list, neutral_list):
    # Load dictionary from a pickle file
    with open("../generate_bert_embeddings/similarities.pkl", "rb") as file:
        loaded_dict = pickle.load(file)

    print(target_list)

    means = []
    years = ['06', '07', '08', '09', '10', '11', '12']
    for year in years:

        v1, v2 = [], []
        count = 0

        for word in target_list:
            vectors = []
            if '{}_{}'.format(word, year) in embeddings_nahar:
                v1.append(embeddings_nahar['{}_{}'.format(word, year)])
                # print(v1)
                print('yes')
                # print(v1.shape)
            else:
                if len(word.split(" ")) > 1:
                    # vectors = []
                    for wt in word.split(" "):
                        possible = loaded_dict[wt]
                        print(possible)
                        for p in possible:
                            if '{}_{}'.format(p, year) in embeddings_nahar:
                                vectors.append(embeddings_nahar['{}_{}'.format(p, year)])
                                print('yes')
                                # print(embeddings_nahar['{}_{}'.format(year, p)][:10])
                            else:
                                tokenized_word = tokenizer.tokenize(p)

                                if not tokenized_word:
                                    print(f"The word '{p}' could not be tokenized.")
                                    continue

                                # Check if any of the tokens are in the vocab_vectors
                                for token in tokenized_word:
                                    # Check if the token exists in the vocab_vectors
                                    token_key = token.replace('##', '')  # Remove '##' if it exists for subwords
                                    if f"{token_key}_{year}" in embeddings_nahar:  # Add the year if your embeddings are year-specific
                                        vectors.append(embeddings_nahar[f"{token_key}_{year}"])
                                print('yesssssssssssssssssssssssssssssssssss')
                                count += 1
                            # try:
                            #     vectors.append(embeddings_nahar['{}_{}'.format(year, p)])
                            # except:
                            #     count += 1
                else:
                    possible = loaded_dict[word]
                    print(possible)
                    for p in possible:
                        if '{}_{}'.format(p, year) in embeddings_nahar:
                            print('yes')
                            vectors.append(embeddings_nahar['{}_{}'.format(p, year)])
                            # print(embeddings_nahar['{}_{}'.format(year, p)][:10])
                        else:
                            tokenized_word = tokenizer.tokenize(p)

                            if not tokenized_word:
                                print(f"The word '{p}' could not be tokenized.")
                                continue

                            # Check if any of the tokens are in the vocab_vectors
                            for token in tokenized_word:
                                # Check if the token exists in the vocab_vectors
                                token_key = token.replace('##', '')  # Remove '##' if it exists for subwords
                                if f"{token_key}_{year}" in embeddings_nahar:  # Add the year if your embeddings are year-specific
                                    vectors.append(embeddings_nahar[f"{token_key}_{year}"])
                            print('yesssssssssssssssssssssssssssssssssss')
                            count += 1
            if vectors != []:
                final_vector = np.mean(vectors, axis=0)
                v1.append(final_vector)


                # print(f'DID NOT FIND {year}_{word} in embeddings nahar')
        for word in target_list:
            vectors = []
            if '{}_{}'.format(word, year) in embeddings_assafir:
                print('yes')
                v2.append(embeddings_assafir['{}_{}'.format(word, year)])
                # print(v2.shape)
                # print(v2)
            else:
                if len(word.split(" ")) > 1:

                    for wt in word.split(" "):
                        possible = loaded_dict[wt]
                        print(possible)
                        for p in possible:
                            if '{}_{}'.format(p, year) in embeddings_assafir:
                                print('yes')
                                vectors.append(embeddings_assafir['{}_{}'.format(p, year)])
                                # print(embeddings_assafir['{}_{}'.format(year, p)][:10])
                            else:
                                tokenized_word = tokenizer.tokenize(p)

                                if not tokenized_word:
                                    print(f"The word '{p}' could not be tokenized.")
                                    continue

                                # Check if any of the tokens are in the vocab_vectors
                                for token in tokenized_word:
                                    # Check if the token exists in the vocab_vectors
                                    token_key = token.replace('##', '')  # Remove '##' if it exists for subwords
                                    if f"{token_key}_{year}" in embeddings_assafir:  # Add the year if your embeddings are year-specific
                                        vectors.append(embeddings_assafir[f"{token_key}_{year}"])
                                count += 1
                else:
                    vectors = []
                    possible = loaded_dict[word]
                    print(possible)
                    for p in possible:
                        if '{}_{}'.format(p, year) in embeddings_assafir:
                            vectors.append(embeddings_assafir['{}_{}'.format(p, year)])
                            print('yes')
                            # print(embeddings_assafir['{}_{}'.format(year, p)][:10])
                        else:
                            tokenized_word = tokenizer.tokenize(p)

                            if not tokenized_word:
                                print(f"The word '{p}' could not be tokenized.")
                                continue

                            # Check if any of the tokens are in the vocab_vectors
                            for token in tokenized_word:
                                # Check if the token exists in the vocab_vectors
                                token_key = token.replace('##', '')  # Remove '##' if it exists for subwords
                                if f"{token_key}_{year}" in embeddings_assafir:  # Add the year if your embeddings are year-specific
                                    vectors.append(embeddings_assafir[f"{token_key}_{year}"])
                            count += 1

            if vectors != []:
                final_vector = np.mean(vectors, axis=0)
                print(final_vector[:10])
                v2.append(final_vector)
                # print(f'DID NOT FIND {year}_{word} in embeddings assafir')

        # print(F'DID NOT FIND {count} words')
        v1 = np.mean(v1, axis=0)
        v2 = np.mean(v2, axis=0)

        print(v1.shape, v1)
        print(v2.shape, v2)

        # v1 = np.mean([embeddings_nahar['{}_{}'.format(year, word)] for word in target_list if '{}_{}'.format(year, word) in embeddings_nahar], axis=0)
        # v2 = np.mean([embeddings_assafir['{}_{}'.format(year, word)] for word in target_list if '{}_{}'.format(year, word) in embeddings_assafir], axis=0)

        x = []
        y = []
        for word in neutral_list:
            try:
                x.append(calc_distance_between_vectors(v1, get_word_embedding(word=word, tokenizer=tokenizer), distype='cossim'))
                y.append(calc_distance_between_vectors(v2, get_word_embedding(word=word, tokenizer=tokenizer), distype='cossim'))
            except:
                pass
        C = [x_ - y_ for x_, y_ in zip(x, y)]
        print(C)
        # values.append(C)
        means.append(np.mean(C))
        # bounds.append(pb.bootstrap(C, confidence=0.95, iterations=1000, sample_size=.9, statistic=np.mean))
    # return values, means, bounds
    return means


if __name__ == '__main__':
    entities_target_neutral_lists = {
        'Arab World Slacktivism': {
            'entities': '../generate_bert_embeddings/entities/bias_quantification/entities1.txt',
            'neutral_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_1.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_1.txt',
            ]
        },
        'Bashir Gemayel Eelection': {
            'entities': '../generate_bert_embeddings/entities/bias_quantification/entities2.txt',
            'neutral_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_2.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_2.txt',
            ]
        },
        'Israeli Invasion': {
            'entities': '../generate_bert_embeddings/entities/bias_quantification/entities3.txt',
            'neutral_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_3.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_3.txt',
            ]
        },
        'Palestinians': {
            'entities': '../generate_bert_embeddings/entities/bias_quantification/entities4.txt',
            'neutral_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_4.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_4.txt',
            ]
        },
        'Philip Habib': {
            'entities': '../generate_bert_embeddings/entities/bias_quantification/entities5.txt',
            'neutral_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_5.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_5.txt',
            ]
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UBC-NLP-MARBERTv2', help='name of the model for which embeddings are stored')
    args = parser.parse_args()

    # Load the model and tokenizer
    model_name = args.model_name
    path_to_model = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/{}/".format(model_name)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    model = AutoModelForMaskedLM.from_pretrained(path_to_model, output_hidden_states=True).cuda()
    model.eval()

    path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/'.format(args.model_name)
    path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/'.format(args.model_name)

    with open(os.path.join(path_nahar, 'words_per_year.pickle'), 'rb') as handle:
        embeddings_nahar = pickle.load(handle)
        print(len(embeddings_nahar))

    with open(os.path.join(path_assafir, 'words_per_year.pickle'), 'rb') as handle:
        embeddings_assafir = pickle.load(handle)
        print(len(embeddings_assafir))

    # read each word
    for event_name in entities_target_neutral_lists:
        print(f"Processing {event_name}")
        file_entities = entities_target_neutral_lists[event_name]['entities']
        file_negconn = entities_target_neutral_lists[event_name]['neutral_list'][0]
        file_posconn = entities_target_neutral_lists[event_name]['neutral_list'][1]

        with open(file_entities, encoding='utf-8') as f:
            entities = f.readlines()
            entities = [e.strip().replace("\n", "") for e in entities]

        with open(file_negconn, encoding='utf-8') as f:
            negconn = f.readlines()
            negconn = [e.strip().replace("\n", "") for e in negconn]

        with open(file_posconn, encoding='utf-8') as f:
            posconn = f.readlines()
            posconn = [e.strip().replace("\n", "") for e in posconn]

        neutral_list = negconn + posconn

        biases_event = embedding_bias(embeddings_nahar, embeddings_assafir, target_list=entities, neutral_list=neutral_list)


        plot_embedding_bias_over_time(biases=biases_event,
                                      ylabel=f"Avg. Embedding Bias for {event_name}",
                                      save_dir="plots/",
                                      fig_name=f"{event_name}")
