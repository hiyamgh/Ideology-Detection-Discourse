import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import argparse
from prompts import prompts_dict


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust as needed


def get_response(prompt):
    text = prompt
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# mapping of the events to the first 3 days of the file names
events = {
    'Philip_Habib_Negotiations': ['820607', '820608', '820609'],
    'Alexander_Haig_resignation': ['820625', '820626', '820627'],
    'The_PLO_approves_Philip Habibs_initiative_to_withdraw_from_Lebanon': ['820807', '820808', '820809'],
    'Bachir_Gemayel_Election': ['820823', '820824', '820825'],
    'Arafat_and_the_PLO_withdraw_from_Beirut': ['820914', '820915', '820916'],
    'Sabra_and_Shatila_Massacre': ['820915', '820916', '820917'],
    'Election_of_Amine_Gemayel': ['820921', '820922', '820923']
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLMs for Hate & Sectarian Speech Detection")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct/", help="The model to use for prediction")
    parser.add_argument("--event_name", type=str, default="Bachir_Gemayel_Election")
    parser.add_argument("--discourse_structure", type=str, default="modality")
    parser.add_argument("--template_name", type=str, default="direct")
    parser.add_argument("--prompt", type=str, default="Hello, who are you and what do you do?")
    # parser.add_argument("--prompt_file", type=str, default="prompts_updated/prompt9.txt", help="The model to use for prediction")
    # parser.add_argument("--entity_name", type=str, default="اسرائيل", help="The name of the entity to extract connotations for")
    args = parser.parse_args()

    # rootdir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/'
    rootdir = "/scratch/8379933-hg31/txt_files/"
    event_name = args.event_name.strip()
    # rootdir = '../generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/'

    archives = ['An-Nahar', 'As-Safir']

    final_chunks = {}
    for archive in archives:
        final_chunks[archive] = []
        path = os.path.join(rootdir, archive)
        for file in os.listdir(path):
            if '.txt' in file and any([substr in file for substr in events[event_name]]):
                with open(os.path.join(path, file), encoding='utf-8') as f:
                    lines = f.readlines()
                    chunks = [lines[i: i + 5] for i in range(0, len(lines), 5)]
                    for c in chunks:
                        fc = '\n'.join([l.replace('\n', '') for l in c])
                        # print(file, fc)
                        final_chunks[archive].append(fc)

    login(token="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")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    suffix = "/scratch/shared/ai/models/llms/hugging_face"
    model_path = os.path.join(suffix, args.model_name)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 force_download=True,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_name = args.model_name.replace("/", "-")
    template = args.template_name
    discourse_structure = args.discourse_structure

    prompt_original = prompts_dict[event_name][discourse_structure][template]

    for archive in archives:
        output_dir = f'Output-Discourse/{archive}-{model_name}-{event_name}-{template}-{discourse_structure}/'
        mkdir(output_dir)

        for text_chunk in final_chunks[archive]:
            # with open(args.prompt_file, encoding="utf-8") as f:
                # prompt = f.read().replace('{entity_name}', f'{args.entity_name}').replace("{text}", f"{text_chunk}")

            prompt = prompt_original.replace("{text}", f"{text_chunk}")

            print(prompt)
            response = get_response(prompt)
            print(response)
            print('===================================================\n\n')
            with open(os.path.join(output_dir, 'responses.txt'), 'a', encoding='utf-8') as f:
                f.write(prompt)
                f.write(response + '\n')
                f.write('===================================================\n\n')
            f.close()
