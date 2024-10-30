import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import argparse


def get_response(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=input_len + 20,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)) + 1:]  # Skip the prompt length

    return response.strip()  # Strip any leading/trailing whitespace


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# mapping of the events to the first 3 days of the file names
events = {
    'Philip Habib Negotiations': ['820607', '820608', '820609'],
    'Alexander Haig resignation': ['820625', '820626', '820627'],
    'The PLO approves Philip Habibs initiative to withdraw from Lebanon': ['820807', '820808', '820809'],
    'Bachir Gemayel Election': ['820823', '820824', '820825'],
    'Arafat and the PLO withdraw from Beirut': ['820914', '820915', '820916'],
    'Sabra and Shatila Massacre': ['820915', '820916', '820917'],
    'Election of Amine Gemayel': ['820921', '820922', '820923']
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLMs for Hate & Sectarian Speech Detection")
    parser.add_argument("--model_name", type=str, default="core42/jais-13b", help="The model to use for prediction")
    parser.add_argument("--prompt_file", type=str, default="prompts/prompt0.txt", help="The model to use for prediction")
    args = parser.parse_args()

    rootdir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/'
    archives = ['An-Nahar', 'As-Safir']

    final_chunks = {}
    for archive in archives:
        final_chunks[archive] = {}
        for event in events:
            final_chunks[archive][event] = []
            path = os.path.join(rootdir, archive)
            for file in os.listdir(path):
                if '.txt' in file and any([substr in file for substr in events[event]]):
                    with open(os.path.join(path, file), encoding='utf-8') as f:
                        lines = f.readlines()
                        chunks = [lines[i: i + 4] for i in range(0, len(lines), 4)]
                        for c in chunks:
                            fc = '\n'.join([l.replace('\n', '') for l in c])
                            print(file, fc)
                            final_chunks[archive][event].append(fc)

    login(token="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")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "{}".format(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    for archive in archives:
        for event in events:
            output_dir = 'output/{}/{}/{}-{}/'.format(archive, event, args.model_name.replace("/", "-"), args.prompt_file.replace("prompts/", "").replace(".txt", ""))
            mkdir(output_dir)

            for text_chunk in final_chunks[archive][event]:
                with open(args.prompt_file, encoding="utf-8") as f:
                    prompt = f.read().format(text_chunk)

                response = get_response(prompt)
                print(response)
                with open(os.path.join(output_dir, 'output.txt'), 'a', encoding='utf-8') as f:
                    f.write(response + '\n')
                    f.write('===================================================\n\n')
                f.close()
