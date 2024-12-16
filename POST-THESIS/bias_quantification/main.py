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
from datetime import date, datetime
from scipy.stats import wasserstein_distance


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


def plot_embedding_bias_over_time(biases, years, ylabel, event_name, distance_type, save_dir):
    # Step 1: Create a unified list of years
    unified_years = sorted(set(years[0]).union(set(years[1])))

    for archive_name in biases:
        print(biases[archive_name])

        aligned_biases = []
        archive_years = years[0] if archive_name == "An-Nahar" else years[1]
        year_to_bias = dict(zip(archive_years, biases[archive_name]))

        for year in unified_years:
            aligned_biases.append(year_to_bias.get(year, None))  # None if year is missing

        # Filter out None values for plotting
        valid_indices = [i for i, b in enumerate(aligned_biases) if b is not None]
        valid_biases = [aligned_biases[i] for i in valid_indices]

        # plt.plot(list(range(len(biases[model_name][archive_name]))), biases[model_name][archive_name], marker='o', label=f"EB for {archive_name}")
        plt.plot(valid_indices, valid_biases, marker='o', label=f"EB for {archive_name}")

    plt.hlines(y=0, xmin=0, xmax=len(unified_years) - 1, colors='black', linestyles='--', lw=2, label='EB=0')

    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.grid(axis='y')

    # plt.xticks(list(range(len(biases[model_name][archive_name]))), ['1982-{}'.format(m) for m in years], rotation=45)
    plt.xticks(list(range(len(unified_years))), [str(year) for year in unified_years], rotation=45)
    plt.ylabel(ylabel)

    mkdir(save_dir)
    fig_name = '{}_{}_{}'.format(event_name, model_name, distance_type)
    plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
    plt.close()


def plot_embedding_bias_over_time_barplots(biases, years, ylabel, event_name, distance_type, save_dir):
    # Step 1: Create a unified list of years
    unified_years = sorted(set(years[0]).union(set(years[1])))

    # Step 2: Define bar width and offsets
    num_archives = len(biases)
    bar_width = 0.4  # Width of each bar
    x_indices = np.arange(len(unified_years))  # x positions for the years
    offsets = np.linspace(-bar_width * (num_archives - 1) / 2, bar_width * (num_archives - 1) / 2, num_archives)

    plt.figure(figsize=(12, 6))

    # Step 3: Plot bars for each archive
    for idx, (archive_name, archive_biases) in enumerate(biases.items()):
        aligned_biases = []
        archive_years = years[0] if archive_name == "An-Nahar" else years[1]
        year_to_bias = dict(zip(archive_years, archive_biases))

        for year in unified_years:
            aligned_biases.append(year_to_bias.get(year, None))  # None if year is missing

        # Filter out None values for plotting
        valid_indices = [i for i, b in enumerate(aligned_biases) if b is not None]
        valid_biases = [aligned_biases[i] for i in valid_indices]

        # Plot bars with offsets for grouping
        plt.bar(
            x_indices[valid_indices] + offsets[idx],
            valid_biases,
            width=bar_width,
            label=f"EB for {archive_name}",
        )

    # Step 4: Add horizontal line, labels, and legend
    plt.hlines(y=0, xmin=-0.5, xmax=len(unified_years) - 0.5, colors='black', linestyles='--', lw=2, label='EB=0')
    plt.xticks(x_indices, [str(year) for year in unified_years], rotation=45)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.legend()

    # Step 5: Save the plot
    mkdir(save_dir)
    fig_name = '{}_{}_{}_barplots'.format(event_name, model_name, distance_type)  # Replace "model_name" with actual variable
    plt.savefig(os.path.join(save_dir, fig_name + '.png'), dpi=300)
    plt.close()


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


def get_week_coverage_sorted(archive, months_to_include=None):
    """
    loops over the .txt files and returns a sorted list of months-weeknbs
    """
    months_weeks = set()
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"

        if months_to_include is not None:
            if month not in months_to_include:
                continue

        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        given_date = date(int(year), int(month), int(day))
        week_number = given_date.isocalendar()[1]

        month_week = f"{month}_{str(week_number)}"

        months_weeks.add(month_week)

    months_weeks_l = list(months_weeks)
    months_weeks_ls = sorted(months_weeks_l, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    print(f"months_weeks sorted for {archive}: {months_weeks_ls}")

    return months_weeks_ls


def get_biweek_coverage_sorted(archive, months_to_include=None):
    """
    loops over the .txt files and returns a sorted list of months-biweeknbs
    """
    months_biweeks = set()
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"

        if months_to_include is not None:
            if month not in months_to_include:
                continue

        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        date = datetime(int(year), int(month), int(day))
        day_of_year = date.timetuple().tm_yday
        biweek_number = (day_of_year - 1) // 14 + 1

        month_biweek = f"{month}_{str(biweek_number)}"
        months_biweeks.add(month_biweek)

    months_biweeks_l = list(months_biweeks)
    months_biweeks_ls = sorted(months_biweeks_l, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    print(f"months_biweeks sorted for {archive}: {months_biweeks_ls}")

    return months_biweeks_ls


def embedding_bias(embeddings, years, target_list1, target_list2, neutral_list, distance_type, tokenizer, model, device="cpu"):
    means = []
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='UBC-NLP-MARBERTv2', help='name of the model for which embeddings are stored')
    parser.add_argument("--split_by", type=str, default="weekly", help="either `monthly` or `weekly`")
    parser.add_argument("--disttype", type=str, default="cossim", help="distance measure")
    args = parser.parse_args()

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
            ],
            'months_to_include': ['08', '09']
        },
        'Israeli Invasion': {
            'neutral_list': '../generate_bert_embeddings/entities/bias_quantification/entities3.txt',
            'target_list': [
                '../generate_bert_embeddings/entities/bias_quantification/negative_conn_3.txt',
                '../generate_bert_embeddings/entities/bias_quantification/positive_conn_3.txt',
            ],
            'months_to_include': ['06', '07', '08', '09']
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

    model_name = args.model_name
    split_by = args.split_by
    dist_type = args.disttype

    if split_by == 'weekly':
        months_weeknbs_nahar = get_week_coverage_sorted(archive="An-Nahar")
        months_weeknbs_assafir = get_week_coverage_sorted(archive="As-Safir")
    elif split_by == 'biweekly':
        months_weeknbs_nahar = get_biweek_coverage_sorted(archive="An-Nahar")
        months_weeknbs_assafir = get_biweek_coverage_sorted(archive="As-Safir")
    else:
        raise ValueError(f"the provided split {split_by} is not supported.")

    for event_name in entities_target_neutral_lists:
        biases = {}

        path_nahar = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/An-Nahar/{}/{}/'.format(model_name, split_by)
        path_assafir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/embeddings/As-Safir/{}/{}/'.format(model_name, split_by)

        path_to_model_nahar = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/An-Nahar/{}/".format(model_name)
        path_to_model_assafir = "/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/As-Safir/{}/".format(model_name)

        # tokenizers for each archive
        tokenizer_nahar = AutoTokenizer.from_pretrained(path_to_model_nahar)
        tokenizer_assafir = AutoTokenizer.from_pretrained(path_to_model_assafir)

        # models for each archive
        model_nahar = AutoModelForMaskedLM.from_pretrained(path_to_model_nahar, output_hidden_states=True).cuda()
        model_nahar.eval()

        model_assafir = AutoModelForMaskedLM.from_pretrained(path_to_model_assafir,
                                                             output_hidden_states=True).cuda()
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

        if 'months_to_include' in entities_target_neutral_lists[event_name]:
            if split_by == 'weekly':
                months_weeknbs_nahar = get_week_coverage_sorted(archive="An-Nahar", months_to_include=entities_target_neutral_lists[event_name]['months_to_include'])
                months_weeknbs_assafir = get_week_coverage_sorted(archive="As-Safir", months_to_include=entities_target_neutral_lists[event_name]['months_to_include'])
            elif split_by == 'biweekly':
                months_weeknbs_nahar = get_biweek_coverage_sorted(archive="An-Nahar", months_to_include=entities_target_neutral_lists[event_name]['months_to_include'])
                months_weeknbs_assafir = get_biweek_coverage_sorted(archive="As-Safir", months_to_include=entities_target_neutral_lists[event_name]['months_to_include'])
        else:
            pass

        with open(file_entities, encoding='utf-8') as f:
            entities = f.readlines()
            entities = [e.strip().replace("\n", "") for e in entities]

        with open(file_negconn, encoding='utf-8') as f:
            negconn = f.readlines()
            negconn = [e.strip().replace("\n", "") for e in negconn]

        with open(file_posconn, encoding='utf-8') as f:
            posconn = f.readlines()
            posconn = [e.strip().replace("\n", "") for e in posconn]

        biases_nahar = embedding_bias(embeddings_nahar, years=months_weeknbs_nahar, target_list1=negconn,
                                      target_list2=posconn, neutral_list=entities, distance_type=dist_type,
                                      tokenizer=tokenizer_nahar, model=model_nahar)
        biases['An-Nahar'] = biases_nahar

        biases_assafir = embedding_bias(embeddings_assafir, years=months_weeknbs_assafir, target_list1=negconn,
                                        target_list2=posconn, neutral_list=entities, distance_type=dist_type,
                                        tokenizer=tokenizer_assafir, model=model_assafir)
        biases['As-Safir'] = biases_assafir
        # read each word
        # for event_name in entities_target_neutral_lists:

        plot_embedding_bias_over_time(biases=biases,
                                      years=[months_weeknbs_nahar, months_weeknbs_assafir],
                                      ylabel=f"Avg. Embedding Bias for {event_name}",
                                      event_name=event_name,
                                      distance_type=dist_type,
                                      save_dir="plots/"
                                      )

        plot_embedding_bias_over_time_barplots(biases=biases,
                                      years=[months_weeknbs_nahar, months_weeknbs_assafir],
                                      ylabel=f"Avg. Embedding Bias for {event_name}",
                                      event_name=event_name,
                                      distance_type=dist_type,
                                      save_dir=f"plots/{split_by}"
                                      )