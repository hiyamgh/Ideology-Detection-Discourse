import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import pickle
import gc
import argparse
import os
from datetime import date, datetime


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def tokens_to_batches(ds, tokenizer, batch_size, max_length):
    batches = []
    batch = []
    batch_counter = 0

    print('Dataset: ', ds)
    counter = 0

    for file in ds:
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                counter += 1
                if counter % 1000 == 0:
                    print('Num sentences: ', counter)

                # Treat each line as a separate sentence
                text = line.strip()

                # Add special tokens for BERT
                marked_text = "[CLS] " + text + " [SEP]"
                tokenized_text = tokenizer.tokenize(marked_text)

                # Divide into chunks of max_length if needed
                for i in range(0, len(tokenized_text), max_length):
                    batch_counter += 1
                    input_sequence = tokenized_text[i:i + max_length]
                    indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)
                    batch.append((indexed_tokens, input_sequence))

                    # Add batch to batches list when batch size is reached
                    if batch_counter % batch_size == 0:
                        batches.append(batch)
                        batch = []

    # Add any remaining batch
    if batch:
        batches.append(batch)

    print('\nTokenization done!')
    print('len batches: ', len(batches))

    return batches


def get_token_embeddings(batches, model, batch_size):

    token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)

        actual_batch_size = len(batch)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)

        # lens = [len(x[0]) for x in batch]
        # max_len = max(lens)

        tokens_tensor = torch.zeros(actual_batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(actual_batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(actual_batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][-4:] #last four layers of the encoder

        for batch_i in range(actual_batch_size):

            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                # Holds last 4 layers of hidden states for each token
                hidden_layers = []

                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                token_embeddings.append(hidden_layers)
                tokenized_text.append(batch_tokens[batch_i][token_i])

    return token_embeddings, tokenized_text


def average_save_and_print(vocab_vectors, save_path):
    for k, v in vocab_vectors.items():

        if len(v) == 2:
            avg = v[0] / v[1]
            vocab_vectors[k] = avg

    with open(save_path, 'wb') as handle:
        pickle.dump(vocab_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length):

    vocab_vectors = {}
    vocab_vectors_avg = {}

    for month in datasets:

        all_batches = tokens_to_batches(datasets[month], tokenizer, batch_size, max_length)
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            splitted_array = np.zeros((1, 768))
            prev_token = ""
            prev_array = np.zeros((1, 768))

            for i, token_i in enumerate(tokenized_text):

                array = token_embeddings[i]

                if token_i.startswith('##'):

                    if prev_token:
                        splitted_tokens.append(prev_token)
                        prev_token = ""
                        splitted_array = prev_array

                    splitted_tokens.append(token_i)
                    splitted_array += array

                else:

                    if token_i + '_' + month in vocab_vectors:
                        vocab_vectors[token_i + '_' + month][0] += array
                        vocab_vectors[token_i + '_' + month][1] += 1
                    else:
                        vocab_vectors[token_i + '_' + month] = [array, 1]

                    if splitted_tokens:
                        sarray = splitted_array / len(splitted_tokens)
                        stoken_i = "".join(splitted_tokens).replace('##', '')

                        if stoken_i + '_' + month in vocab_vectors:
                            vocab_vectors[stoken_i + '_' + month][0] += sarray
                            vocab_vectors[stoken_i + '_' + month][1] += 1
                        else:
                            vocab_vectors[stoken_i + '_' + month] = [sarray, 1]

                        splitted_tokens = []
                        splitted_array = np.zeros((1, 768))

                    prev_array = array
                    prev_token = token_i

            del token_embeddings
            del tokenized_text
            del batches
            gc.collect()

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    average_save_and_print(vocab_vectors, embeddings_path)
    del vocab_vectors
    del vocab_vectors_avg
    gc.collect()


def get_datasets_by_month(archive_name):
    datasets = {}
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive_name)
    for file in os.listdir(root_dir):
        month = file[2:4]
        if month not in datasets:
            datasets[month] = [os.path.join(root_dir, file)]
        else:
            datasets[month].append(os.path.join(root_dir, file))
    return datasets


def get_datasets_by_week(archive_name):
    datasets = {}
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(
        archive_name)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"
        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        given_date = date(int(year), int(month), int(day))
        week_number = given_date.isocalendar()[1]

        month_week = f"{month}_{str(week_number)}"

        if month_week not in datasets:
            datasets[month_week] = [os.path.join(root_dir, file)]
        else:
            datasets[month_week].append(os.path.join(root_dir, file))
    return datasets


def get_datasets_by_biweek(archive_name):
    datasets = {}
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/{}/'.format(archive_name)
    for file in os.listdir(root_dir):
        month = file[2:4]
        day = file[4:6]
        year = "1982"
        print(f'processing file {file} with year: {year}, month: {month}, day: {day}')
        date = datetime(int(year), int(month), int(day))
        day_of_year = date.timetuple().tm_yday
        biweek_number = (day_of_year - 1) // 14 + 1

        month_biweek = f"{month}_{str(biweek_number)}"

        if month_biweek not in datasets:
            datasets[month_biweek] = [os.path.join(root_dir, file)]
        else:
            datasets[month_biweek].append(os.path.join(root_dir, file))
    return datasets


def get_datasets_by_year(archive_name, year):
    datasets = {}
    root_dir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/{}/txt_files/{}/'.format(year, archive_name)
    for file in os.listdir(root_dir):
        year = f"{year}"
        print(f'processing file {file}')

        if year not in datasets:
            datasets[year] = [os.path.join(root_dir, file)]
        else:
            datasets[year].append(os.path.join(root_dir, file))
    return datasets


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument('--path_to_model', type=str, help='Paths to the fine-tuned BERT model')
    parser.add_argument('--split_by', type=str, help="To split the time specific embeddings. "
                                                     "Value must be `monthly`, `weekly`, `biweekly`, or `yearly`")
    parser.add_argument("--year", type=str, help="The year to get teh dataset from, if `--split_by` is set to `yearly`")
    parser.add_argument('--archive', type=str, help='name of the archive to get embeddings for', default='An-Nahar')
    args = parser.parse_args()

    split_by = args.split_by
    year = args.year
    if split_by == "monthly":
        datasets = get_datasets_by_month(args.archive)
    elif split_by == "weekly":
        datasets = get_datasets_by_week(args.archive)
    elif split_by == "biweekly":
        datasets = get_datasets_by_biweek(args.archive)
    elif split_by == "yearly":
        datasets = get_datasets_by_year(args.archive, year)
    else:
        raise ValueError(f"Value of `split_by` argument must be either `monthly`, `weekly`, `biweekly`, or `yearly`, you provided {split_by}")

    tokenizer = AutoTokenizer.from_pretrained(args.path_to_model)
    model = AutoModelForMaskedLM.from_pretrained(args.path_to_model, output_hidden_states=True)

    model.cuda()
    model.eval()

    model_name = args.path_to_model.replace('/onyx/data/p118/POST-THESIS/generate_bert_embeddings/trained_models/', "").replace("An-Nahar", "").replace("As-Safir", "").replace("/", "")

    # create a directory to save time-specific embeddings for words/tokens,
    # inspired by archive name, model name, and the `split_by`
    save_dir = 'opinionated_articles_DrNabil/{}/embeddings/{}/{}/{}'.format(year, args.archive, model_name, split_by)
    mkdir(save_dir)
    embeddings_path = os.path.join(save_dir, "embeddings.pickle")

    get_time_embeddings(embeddings_path, datasets, tokenizer, model, args.batch_size, args.max_length)
