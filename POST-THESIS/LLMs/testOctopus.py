from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os


def get_response(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # print(input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # Create attention mask
    attention_mask = attention_mask
    # print(attention_mask)
    inputs = input_ids
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        attention_mask=attention_mask,  # Pass attention mask here
        top_p=0.85,
        temperature=0.7,
        max_length=input_len + 10,
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
    'Philip-Habib-Negotiations': ['820607', '820608', '820609'],
    'Alexander-Haig-resignation': ['820625', '820626', '820627'],
    'The-PLO-approves-Philip-Habibs-initiative-to-withdraw-from-Lebanon': ['820807', '820808', '820809'],
    'Bachir-Gemayel-Election': ['820823', '820824', '820825'],
    'Bachir-Gemayel-Assassination': ['820915', '820916', '820917'],
    'Arafat-and-the-PLO-withdraw-from-Beirut': ['820914', '820915', '820916'],
    'Sabra-and-Shatila-Massacre': ['820915', '820916', '820917'],
    'Election-of-Amine-Gemayel': ['820921', '820922', '820923']
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prompting LLMs")
    parser.add_argument("--model_checkpoint", type=str, default="inceptionai/jais-30b-chat-v3", help="The model to use for prediction")
    parser.add_argument("--event_name", type=str, help="The name of the event of interest",  default="Bachir-Gemayel-Assassination")
    parser.add_argument("--prompt_file", type=str, default="prompts_updated/prompt4.txt", help="The prompt file")
    args = parser.parse_args()

    model_name = "/scratch/shared/ai/models/llms/hugging_face/core24/jais-13b-chat/"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    rootdir = 'txt_files/'
    archives = ['An-Nahar', 'As-Safir']
    event = args.event_name

    final_chunks = {}
    for archive in archives:
        final_chunks[archive] = []
        path = os.path.join(rootdir, archive)
        for file in os.listdir(path):
            if '.txt' in file and any([substr in file for substr in events[event]]):
                with open(os.path.join(path, file), encoding='utf-8') as f:
                    lines = f.readlines()
                    chunks = [lines[i: i + 20] for i in range(0, len(lines), 20)]
                    for c in chunks:
                        fc = '\n'.join([l.replace('\n', '') for l in c])
                        # print(file, fc)
                        final_chunks[archive].append(fc)

    output_dir = 'LLM-output/'
    mkdir(output_dir)
    for archive in archives:
        with open(os.path.join(output_dir, f"{archive}-{event}.txt"), "a") as f:
            for text_chunk in final_chunks[archive]:
                with open(args.prompt_file, "r", encoding="utf-8", ) as fin:
                    prompt = fin.read().format(text_chunk)
                    answer = get_response(text=prompt)

                    print(answer)
                    f.write(answer + '\n')

                    print('===========================================================')
                    f.write('=========================================================\n')

    # for prompt in tqdm(prompts):
    #     answer = get_response(text=prompt)
    #     print(answer)
    #     print('===========================================================')
