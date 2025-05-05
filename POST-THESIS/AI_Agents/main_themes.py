import re
import os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import time
from tqdm import tqdm
from crewai.flow.flow import Flow, start, listen
# from pydantic import BaseModel
# from typing import List
# from typesCaseStudy import *
# from crewai.flow.flow import Flow, listen, start
# from case_study_crew import CaseStudyCrew

load_dotenv()

my_llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-1.5-flash",
)

text_nahar = ""
with open("2006-mod/training_file/An-Nahar/2006_An-Nahar.txt", "r", encoding="utf-8") as f:
    file_content = f.read()
    text_nahar += file_content
f.close()

text_asafir = ""
with open("2006-mod/training_file/As-Safir/2006_As-Safir.txt", "r", encoding="utf-8") as f:
    file_content = f.read()
    text_nahar += file_content
f.close()


# Use Path for file locations
current_dir = Path.cwd()
agents_config_path = current_dir / "config" / "agents.yaml"
tasks_config_path = current_dir / "config" / "tasks.yaml"

# Load YAML configuration files
with open(agents_config_path, "r") as file:
    agents_config = yaml.safe_load(file)

with open(tasks_config_path, "r") as file:
    tasks_config = yaml.safe_load(file)

agency_agent = Agent(
    config=agents_config["themes_detection_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

# Filter out spam and vulgar posts
task0 = Task(
    description=tasks_config["identify_themes"]["description"],
    expected_output=tasks_config["identify_themes"]["expected_output"],
    agent=agency_agent,
    # output_file='outputs/themes.md',
    # create_directory=True
)

crew = Crew(
    agents=[agency_agent],
    tasks=[task0],
    verbose=True,
    process=Process.sequential,
    llm=my_llm
)


themes = {
    "position towards Israel": [
        "local and international politics regarding Israel",
        "the destruction of Lebanon's infrastructure by Israel",
        "the exile and displacement of the Lebanese people because of Israel",
        "the Israeli terrorism",
        "Resolution 1559 and its application",
        "the pretext of two kidnapped soldiers",
        "Israeli military arsenal and the asymmetrical war",
    ],
    "position towards Ara Countries and major powers": [
        "Arab and international political class supporting Israel",
        "harsh criticism of regimes and the United Nations",
    ],
}

def split_into_chunks(text, chunk_size=10000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


mkdir("outputs/As-Safir/")
mkdir("outputs/An-Nahar/")

for major_theme in themes:
    for sub_theme in themes[major_theme]:
        nahar_chunked = split_into_chunks(text=text_nahar)
        all_output = ""
        for chunk in nahar_chunked:
            result = crew.kickoff(inputs={"text": chunk, "theme": sub_theme})
            task_output = task0.output

            all_output += task_output.raw

            with open("outputs/An-Nahar/theme_results.md", "a", encoding="utf-8") as f:
                f.write(f"### Results for the theme: {sub_theme}")
                f.write(f"{all_output}")
                f.write('\n\n')

for major_theme in themes:
    for sub_theme in themes[major_theme]:
        nahar_chunked = split_into_chunks(text=text_asafir)
        all_output = ""
        for chunk in nahar_chunked:
            result = crew.kickoff(inputs={"text": chunk, "theme": sub_theme})
            task_output = task0.output

            all_output += task_output.raw

            with open("outputs/As-Safir/theme_results.md", "a", encoding="utf-8") as f:
                f.write(f"### Results for the theme: {sub_theme}")
                f.write(f"{all_output}")
                f.write('\n\n')

            time.sleep(180)


# class ExampleFlow(Flow):
#     @start
#     def generate_case_study_outline(self):
#         print("Kickoff the Case Study Outline Crew")
#         output = crew.kickoff(
#                 inputs={"text": text_nahar[:10000], "theme": "local and international politics regarding Israel"}
#         )
#
#         print("===================== end result from crew ===================================")
#         print(output)
#         print("===================== score ==================================================")
#         return output
#
#     @listen(generate_case_study_outline)
#     def conclude(self, result):
#         return result

# if __name__ == '__main__':
#     flow = ExampleFlow()
#     result = flow.kickoff()
#
#     print(f"Generated fun fact: {result}")


# class CaseStudyState(BaseModel):
#     title: str = "The 1982 Lebanon War"
#     id: str = "oneone"
#     # theme_outline: List[ThemeOutline]
#
#
#
# class CaseStudyFlow(Flow[CaseStudyState]):
#     initial_state = CaseStudyState
#
    # @start
    # def generate_case_study_outline(self):
    #
    #     print("Kickoff the Case Stud Outline Crew")
    #     output = (
    #         crew.kickoff(inputs={"text": text_nahar[:10000], "theme": "local and international politics regarding Israel"})
    #     )
    #
    #     print("===================== end result from crew ===================================")
    #     print(output)
    #     print("===================== score ==================================================")
    #     return output
    #
    # @listen(generate_case_study_outline)
    # def conclude(self, result):
    #     return result



# if __name__ == '__main__':
#     poem_flow = CaseStudyFlow()
#     poem_flow.kickoff()

