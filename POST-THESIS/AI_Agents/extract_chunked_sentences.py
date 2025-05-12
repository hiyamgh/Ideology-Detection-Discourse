import os
from google import genai
from google.genai import types
from instructions_discourse_structures import *
import time
from tqdm import tqdm
import argparse
import json
import time

client = genai.Client(api_key="AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs")


def split_into_chunks(text, chunk_size=10000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def read_document(archive_name):
    text = ""
    with open(os.path.join(f"../LLMs/txt_files/2006/{archive_name}/", f"2006_{archive_name}.txt"), "r", encoding="utf-8") as f:
        file_content = f.read()
        text += file_content
    f.close()
    return text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_name", type=str, default="An-Nahar", help="Name of teh archive, either An-Nahar or As-Safir")
    parser.add_argument("--chunk_size", type=int, default=500, help="chunk size to divide the archive document")
    args = parser.parse_args()

    archive_name = args.archive_name
    cs = args.chunk_size

    doc_text = read_document(archive_name=archive_name)
    text_splitted = split_into_chunks(doc_text, chunk_size=cs)

    print(f"Total number of split text is {len(text_splitted)} for the {archive_name} archive")

    responses = []
    save_dir = f"outputs/{archive_name}/chunked_sentences/"
    mkdir(folder_name=save_dir)
    output_file = "responses.json"

    t1 = time.time()
    for idx, txt in enumerate(text_splitted):
        prompt = extract_sentences_instruction.replace("{text}", txt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        print(response.text)

        print(f"Response {idx + 1}: {response.text}")
        responses.append({
            "chunk_index": idx,
            "prompt": prompt,
            "response": response.text
        })

        with open(os.path.join(save_dir, output_file), "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)

        # # print(response.usage_metadata)
        #
        # # prompt_extract_agency = extract_agency.replace("{sentences}", response.text) # Zero-shot
        # # prompt_extract_agency = extract_agency_fewshot.replace("{sentences}", response.text)  # Few-shot
        # # prompt_extract_agency = extract_agency.replace("{sentences}", response.text)  # Zero-shot
        #
        # # prompt_extract_victimization = extract_victimization.replace("{sentences}", response.text)  # Zero-shot
        # # prompt_extract_nsf = extract_national_self_glorification.replace("{sentences}", response.text)  # Zero-shot
        # # prompt_extract_dramatization = extract_dramatization.replace("{sentences}", response.text)  # Zero-shot
        # # prompt_extract_disclaimer = extract_disclaimer.replace("{sentences}", response.text)  # Zero-shot
        # # prompt_extract_denomination = extract_denomination.replace("{sentences}", response.text)  # Zero-shot
        # prompt_extract_LDC = extract_LDC.replace("{sentences}", response.text)  # Zero-shot
        #
        # # print(prompt_extract_nsf)
        # response = client.models.generate_content(
        #     model="gemini-2.0-flash",
        #     contents=[prompt_extract_LDC]
        # )
        # print(response.text)
        # print(response.usage_metadata)
        print('===========================================================')

        time.sleep(10)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    t2 = time.time()

    print(f"All responses saved to {os.path.join(save_dir, output_file)}")
    print(f"Time taken: {(t2-t1)/60} mins")
