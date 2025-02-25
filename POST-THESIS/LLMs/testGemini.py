import os
from google import genai
from google.genai import types
from instructions import *

client = genai.Client(api_key="AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs")

text_nahar = ""
for file in os.listdir("txt_files/An-Nahar"):
    if file.startswith("820915") or file.startswith("820916"):
        with open(os.path.join("txt_files/An-Nahar/", file), "r", encoding="utf-8") as f:
            file_content = f.read()
            text_nahar += file_content
        f.close()

text_assafir = ""
for file in os.listdir("txt_files/As-Safir"):
    if file.startswith("820915") or file.startswith("820916"):
        with open(os.path.join("txt_files/As-Safir/", file), "r", encoding="utf-8") as f:
            file_content = f.read()
            text_assafir += file_content
        f.close()


# prompt = f"""{text_assafir}"""
prompt = f"""{text_nahar[:10000]}"""
print(prompt)

print(len(text_assafir.split(" ")))

# print("total_tokens: ", client.models.count_tokens(text_assafir))
#
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=instruction_dramatization),
    contents=[prompt]
)


print(response.text)
print(response.usage_metadata)