import os
from google import genai
from google.genai import types
from instructions_themes import *
import time

client = genai.Client(api_key="AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs")


def split_into_chunks(text, chunk_size=10000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


with open("2006_An-Nahar.txt", "r", encoding="utf-8") as f:
    text_nahar = f.read()
f.close()

text_splitted = split_into_chunks(text_nahar, chunk_size=500)

# these are actually the sub-themes
themes = [
    "local and international politics regarding Israel",
    "the destruction of Lebanon's infrastructure by Israel",
    "the exile and displacement of the Lebanese people because of Israel",
    "the Israeli terrorism",
    "Resolution 1559 and its application",
    "the pretext of two kidnapped soldiers",
    "Israeli military arsenal and the asymmetrical war",
]

print()
for theme in themes:
    for txt in text_splitted[:10]:
        prompt = extract_sentences_instruction.replace("{text}", txt)

        print(prompt)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )

        print(response.text)
        print(response.usage_metadata)

        prompt_extract_themes = extract_themes_from_sentences.replace("{theme}", theme).replace("{sentences}", response.text)
        print(prompt_extract_themes)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt_extract_themes]
        )
        print(response.text)
        print(response.usage_metadata)
        print('===========================================================')

        time.sleep(10)
