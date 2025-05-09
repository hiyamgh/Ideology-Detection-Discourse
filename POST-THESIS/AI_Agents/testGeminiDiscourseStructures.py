import os
from google import genai
from google.genai import types
from instructions_discourse_structures import *
import time
from tqdm import tqdm

client = genai.Client(api_key="AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs")


def split_into_chunks(text, chunk_size=10000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# text_nahar = ""
# for file in tqdm(os.listdir("../LLMs/txt_files/An-Nahar"), desc="Gathering files from An-Nahar"):
#     if file.startswith("820915") or file.startswith("820916"):
#         with open(os.path.join("../LLMs/txt_files/An-Nahar/", file), "r", encoding="utf-8") as f:
#             file_content = f.read()
#             text_nahar += file_content
#         f.close()

text_nahar = ""
with open(os.path.join("../LLMs/txt_files/2006/An-Nahar/", "2006_An-Nahar.txt"), "r", encoding="utf-8") as f:
    file_content = f.read()
    text_nahar += file_content
f.close()


text_splitted = split_into_chunks(text_nahar, chunk_size=500)
print(f"Total number of split text: {len(text_splitted)}")

for txt in text_splitted[:10]:
    prompt = extract_sentences_instruction.replace("{text}", txt)

    # print(prompt)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )

    print(response.text)
    # print(response.usage_metadata)

    # prompt_extract_agency = extract_agency.replace("{sentences}", response.text) # Zero-shot
    # prompt_extract_agency = extract_agency_fewshot.replace("{sentences}", response.text)  # Few-shot
    # prompt_extract_agency = extract_agency.replace("{sentences}", response.text)  # Zero-shot

    # prompt_extract_victimization = extract_victimization.replace("{sentences}", response.text)  # Zero-shot
    # prompt_extract_nsf = extract_national_self_glorification.replace("{sentences}", response.text)  # Zero-shot
    # prompt_extract_dramatization = extract_dramatization.replace("{sentences}", response.text)  # Zero-shot
    # prompt_extract_disclaimer = extract_disclaimer.replace("{sentences}", response.text)  # Zero-shot
    # prompt_extract_denomination = extract_denomination.replace("{sentences}", response.text)  # Zero-shot
    prompt_extract_LDC = extract_LDC.replace("{sentences}", response.text)  # Zero-shot

    # print(prompt_extract_nsf)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt_extract_LDC]
    )
    print(response.text)
    print(response.usage_metadata)
    print('===========================================================')

    time.sleep(10)
