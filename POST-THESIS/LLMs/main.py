import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust as needed

# def get_response(text):
#     input_ids = tokenizer(text, return_tensors="pt").input_ids
#     # print(input_ids)
#     attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Create attention mask
#     attention_mask = attention_mask
#     # print(attention_mask)
#     inputs = input_ids
#     input_len = inputs.shape[-1]
#     generate_ids = model.generate(
#         inputs,
#         attention_mask=attention_mask,  # Pass attention mask here
#         top_p=0.85,
#         temperature=0.9,
#         max_length=input_len + 10,
#         min_length=input_len + 4,
#         repetition_penalty=1.2,
#         do_sample=True,
#     )
#     response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
#     response = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)) + 1:]  # Skip the prompt length
#
#     return response.strip()  # Strip any leading/trailing whitespace
#


def get_response(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# # mapping of the events to the first 3 days of the file names
# events = {
#     'Philip Habib Negotiations': ['820607', '820608', '820609'],
#     'Alexander Haig resignation': ['820625', '820626', '820627'],
#     'The PLO approves Philip Habibs initiative to withdraw from Lebanon': ['820807', '820808', '820809'],
#     'Bachir Gemayel Election': ['820823', '820824', '820825'],
#     'Arafat and the PLO withdraw from Beirut': ['820914', '820915', '820916'],
#     'Sabra and Shatila Massacre': ['820915', '820916', '820917'],
#     'Election of Amine Gemayel': ['820921', '820922', '820923']
# }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLMs for Hate & Sectarian Speech Detection")
    parser.add_argument("--model_name", type=str, default="/nvme/h/lb21hg1/.cache/huggingface/Qwen2.5-7B-Instruct/", help="The model to use for prediction")
    parser.add_argument("--prompt_file", type=str, default="prompts/prompt7.txt", help="The model to use for prediction")
    parser.add_argument("--entity_name", type=str, default="اسرائيل", help="The name of the entity to extract connotations for")
    args = parser.parse_args()

    rootdir = '/onyx/data/p118/POST-THESIS/generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/'
    # rootdir = '../generate_bert_embeddings/opinionated_articles_DrNabil/1982/txt_files/'

    archives = ['An-Nahar', 'As-Safir']

    # final_chunks = {}
    # for archive in archives:
    #     final_chunks[archive] = {}
    #     for event in events:
    #         final_chunks[archive][event] = []
    #         path = os.path.join(rootdir, archive)
    #         for file in os.listdir(path):
    #             if '.txt' in file and any([substr in file for substr in events[event]]):
    #                 with open(os.path.join(path, file), encoding='utf-8') as f:
    #                     lines = f.readlines()
    #                     chunks = [lines[i: i + 20] for i in range(0, len(lines), 20)]
    #                     for c in chunks:
    #                         fc = '\n'.join([l.replace('\n', '') for l in c])
    #                         # print(file, fc)
    #                         final_chunks[archive][event].append(fc)

    final_chunks = {}
    for archive in archives:
        final_chunks[archive] = []
        path = os.path.join(rootdir, archive)
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding='utf-8') as f:
                lines = f.readlines()
                chunks = [lines[i: i + 10] for i in range(0, len(lines), 10)]
                for c in chunks:
                    fc = '\n'.join([l.replace('\n', '') for l in c])
                    # print(file, fc)
                    final_chunks[archive].append(fc)

    login(token="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")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "{}".format(args.model_name)
    # model_path = "/nvme/h/lb21hg1/.cache/huggingface/Qwen2.5-7B-Instruct/"
    # if 'qwen' in model_path.lower():
    #     tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    # else:


    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 force_download=True,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_name = model_path.replace("/", "-")
    prompt_file = args.prompt_file.replace("prompts/", "").replace(".txt", "")
    for archive in archives:
        output_dir = f'output/{archive}-{model_name}-{prompt_file}/'
        mkdir(output_dir)

        for text_chunk in final_chunks[archive]:
            with open(args.prompt_file, encoding="utf-8") as f:
                prompt = f.read().replace('{entity_name}', f'{args.entity_name}').replace("{text}", f"{text_chunk}")

            print(prompt)
            response = get_response(prompt)
            print(response)
            print('===================================================\n\n')
            with open(os.path.join(output_dir, 'responses.txt'), 'a', encoding='utf-8') as f:
                f.write(prompt)
                f.write(response + '\n')
                f.write('===================================================\n\n')
            f.close()
