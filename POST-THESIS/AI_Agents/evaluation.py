import os
from google import genai
import json
import time
from evaluation_prompts import *
from instructions_discourse_structures import *
import argparse

client = genai.Client(api_key="AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs")


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--discourse_structure", type=str, default="agency",
                        help="name of teh discourse structure of interest")
    parser.add_argument("--archive_name", type=str, default="An-Nahar",
                        help="Name of the archive, either An-Nahar or As-Safir")
    args = parser.parse_args()

    archive_name = args.archive_name
    discourse_structure = args.discourse_structure

    filename = "extracted_sentences.json"
    main_dir = f"outputs/{archive_name}/chunked_sentences/"

    # Load the JSON data
    with open(os.path.join(main_dir, filename), "r", encoding="utf-8") as f:
        sentence_data = json.load(f)

    print(f"Total number of extracted sentences: {len(sentence_data)}")

    batch_size = 5
    batch_results = []

    # for i in range(0, len(sentence_data), batch_size):
    for i in range(0, 10, batch_size):
        batch = sentence_data[i:i + batch_size]
        print(f"\nBatch {i // batch_size + 1}:")

        sentences_template = ""
        for idx, entry in enumerate(batch):
            index = entry["index"]
            s = entry["sentence"].replace("\"", "").replace("\\", "").replace("”", "")
            # sentences_template += f"{i+1}. " + entry["sentence"]
            sentences_template += entry["sentence"] + "\n"
        print(f"Sentence templates:\n\n{sentences_template}")

        # prompt_extract_disc = prompt_template.replace("{sentences}", sentences_template)  # Zero-shot
        prompt = query_analysis_prompt.replace("{query}", extract_agency)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        print(response.text)

        print("//////////////////////////////////////")

        key_question = response.text.split("**Key question:**")[1]

        prompt = document_analysis_prompt.replace("{query}", extract_agency).\
            replace("{query_key_question}", key_question).\
            replace("{sentences}", sentences_template)

        print(prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        print(response.text)
        print("//////////////////////////////////////")

        prompt = judgement_prompt.replace("{query}", extract_agency). \
            replace("{query_key_question}", key_question). \
            replace("{sentences}", sentences_template).\
            replace("{analysis}", response.text)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        print(response.text)

        print('=========================================================================\n\n\n')

#         **Key question:**





