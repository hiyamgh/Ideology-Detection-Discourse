import os
from google import genai
from google.genai import types


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

instruction = """
You are a social scientist applying discourse analysis over Lebanese newspapers.
You will be given text taken from a Lebanese newspaper
Identify sentences that contain either an active or a passive voice:
For each sentence you retrieve, return the following:
If the sentence contains a passive voice, output:
- Voice: <Passive>
- Sentence: <the extracted sentence>
- Passive phrase(s): <the verbs or phrases that were used in the passive form>.
If the sentence contains an active voice, output:
- Voice: <Active>
- Sentence: <the extracted sentence>
- Active Agent(s): <the agent(s) that was the active voice in the sentence>
Provide a justification explaining the reasoning behind your choice.
"""


# prompt = f"""{text_assafir}"""
prompt = f"""{text_nahar}"""
print(prompt)

print(len(text_assafir.split(" ")))

# print("total_tokens: ", client.models.count_tokens(text_assafir))
#
response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=instruction),
    contents=[prompt]
)


print(response.text)
print(response.usage_metadata)