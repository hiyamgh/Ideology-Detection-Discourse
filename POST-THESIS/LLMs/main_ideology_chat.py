import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Adjust as needed


# def get_response(prompt, prompt_ideology, extracted_answer):
#     chat = [
#         {"role": "user", "content": f"You are a helpful social science assistant. Read the text and answer it: {prompt}"},
#         {"role": "assistant", "content": f"My answer is: {extracted_answer}"},
#         {"role": "user", "content": f"If your answer was not 'None' do the following: {prompt_ideology}"},
#     ]
#
#     tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
#
#     outputs = model.generate(tokenized_chat, max_new_tokens=128)
#     print(tokenizer.decode(outputs[0]))
#     return tokenizer.decode(outputs[0])

def get_response(prompt_ideology, extracted_answer):
    chat = [
        {"role": "user", "content": f"You are a helpful social science assistant. Read the extracted sentences and justifcation below: \n{extracted_answer}"},
        {"role": "assistant", "content": f"I have read and understood it. What is your question?"},
        {"role": "user", "content": f"If the content you read was not 'None' do the following: {prompt_ideology}"},
    ]

    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    outputs = model.generate(tokenized_chat, max_new_tokens=128)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])


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
    parser.add_argument("--model_name", type=str, default="/scratch/8379933-hg31/huggingface_models/meta-llama/Llama-3.1-8B-Instruct/", help="The model to use for prediction")
    parser.add_argument("--event_name", type=str, default="Bachir_Gemayel_Election")
    parser.add_argument("--prompt_file", type=str, default="prompts_updated/prompt9.txt", help="The model to use for prediction")
    parser.add_argument("--prompt_ideology", type=str, default="prompts_updated/prompt_ideology.txt", help="The prompt used to extract cvertain ideologies in text")
    parser.add_argument("--entity_name", type=str, default="اسرائيل", help="The name of the entity to extract connotations for")
    args = parser.parse_args()


    event = args.event_name.strip()

    archives = ['An-Nahar', 'As-Safir']


    responses_anahar = "/scratch/8379933-hg31/output/An-Nahar--scratch-8379933-hg31-huggingface_models-meta-llama-Llama-3.1-8B--prompts_updated/prompt9/"
    responses_assafir = "/scratch/8379933-hg31/output/As-Safir--scratch-8379933-hg31-huggingface_models-meta-llama-Llama-3.1-8B--prompts_updated/prompt9/"
    responses_dirs = [responses_anahar, responses_assafir]

    file = args.prompt_ideology
    with open(file, "r") as f:
        prompt_ideology = f.read().strip()

    login(token="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")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model_name

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.float32,
                                                 trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for dir in responses_dirs:
        # with open(os.path.join(dir, "responses_ideology_chat.txt"), "a", encoding="utf-8") as fout:
        with open(os.path.join(dir, "responses_ideology_chatv2.txt"), "a", encoding="utf-8") as fout:
            all_responses_file = os.path.join(dir, "responses.txt")
            with open(all_responses_file, "r", encoding="utf-8") as f:
                prompt_with_answers = f.read().split("===================================================")
                for pwa in prompt_with_answers:
                    # prompt = pwa.split("Answer:")[0].strip()
                    answer = pwa.split("Answer:")[1].strip()
                    # response = get_response(prompt=prompt, prompt_ideology=prompt_ideology, extracted_answer=answer)
                    response = get_response(prompt_ideology=prompt_ideology, extracted_answer=answer)
                    fout.write(response + "\n")
                    fout.write("===============================================\n")
            f.close()
        fout.close()
