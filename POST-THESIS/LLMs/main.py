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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLMs for Hate & Sectarian Speech Detection")
    parser.add_argument("--model_name", type=str, default="core42/jais-13b", help="The model to use for prediction")
    parser.add_argument("--prompt_file", type=str, default="prompts/prompt0.txt", help="The model to use for prediction")
    args = parser.parse_args()

    final_chunks = []
    root_dir = 'txts/'
    for file_name in os.listdir('txts/'):
        if file_name.endswith('.txt'):
            with open(os.path.join(root_dir, file_name), encoding='utf-8') as f:
                lines = f.readlines()
                chunks = [lines[i: i + 4] for i in range(0, len(lines), 4)]
                for chunk in chunks:
                    final_chunks.append('\n'.join([l.replace('\n', '') for l in chunk]))

    login(token="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")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "{}".format(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    for text_chunk in final_chunks:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read().format(text_chunk)

        print(get_response(prompt))
        print('====================================================================')
