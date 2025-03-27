import re
import os
import yaml
from pathlib import Path
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

my_llm = LLM(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini/gemini-1.5-flash",
)

text_nahar = ""
for file in tqdm(os.listdir("txt_files/An-Nahar"), desc="Gathering files from An-Nahar"):
    if file.startswith("820915") or file.startswith("820916"):
        with open(os.path.join("txt_files/An-Nahar/", file), "r", encoding="utf-8") as f:
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

## Define Agents
agency_agent = Agent(
    config=agents_config["agency_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

victimization_agent = Agent(
    config=agents_config["victimization_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

national_self_glorification_agent = Agent(
    config=agents_config["national_self_glorification_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

evidentiality_agent = Agent(
    config=agents_config["evidentiality_agent"], allow_delegation=False, verbose=True, llm=my_llm
)


dramatization_agent = Agent(
    config=agents_config["dramatization_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

disclaimer_agent = Agent(
    config=agents_config["disclaimer_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

denomination_agent = Agent(
    config=agents_config["disclaimer_agent"], allow_delegation=False, verbose=True, llm=my_llm
)

# Filter out spam and vulgar posts
task0 = Task(
    description=tasks_config["identify_agency"]["description"],
    expected_output=tasks_config["identify_agency"]["expected_output"],
    agent=agency_agent,
    output_file='outputs/agency.md',
    create_directory=True
)
# process post with a crew of agents, ultimately delivering a well formatted dialogue
task1 = Task(
    description=tasks_config["identify_victimization"]["description"],
    expected_output=tasks_config["identify_victimization"]["expected_output"],
    agent=victimization_agent,
    output_file='outputs/victimization.md',
    create_directory=True
)

task2 = Task(
    description=tasks_config["identify_national_self_glorification"]["description"],
    expected_output=tasks_config["identify_national_self_glorification"]["expected_output"],
    agent=national_self_glorification_agent,
    output_file='outputs/national_self_glorification.md',
    create_directory=True
)

task3 = Task(
    description=tasks_config["identify_evidentiality"]["description"],
    expected_output=tasks_config["identify_evidentiality"]["expected_output"],
    agent=evidentiality_agent,
    output_file='outputs/evidentiality.md',
    create_directory=True
)

task4 = Task(
    description=tasks_config["identify_dramatization"]["description"],
    expected_output=tasks_config["identify_dramatization"]["expected_output"],
    agent=victimization_agent,
    output_file='outputs/dramatization.md',
    create_directory=True
)

task5 = Task(
    description=tasks_config["identify_disclaimer"]["description"],
    expected_output=tasks_config["identify_disclaimer"]["expected_output"],
    agent=disclaimer_agent,
    output_file='outputs/disclaimer.md',
    create_directory=True
)

task6 = Task(
    description=tasks_config["identify_denomination"]["description"],
    expected_output=tasks_config["identify_denomination"]["expected_output"],
    agent=denomination_agent,
    output_file='outputs/denomination.md',
    create_directory=True
)

# task7 = Task(
#     description=tasks_config["identify_LDC"]["description"],
#     expected_output=tasks_config["identify_LDC"]["expected_output"],
#     agent=LDC,
# )


crew = Crew(
    agents=[agency_agent, victimization_agent, national_self_glorification_agent, evidentiality_agent, dramatization_agent, disclaimer_agent, denomination_agent],
    tasks=[task0, task1, task2, task3, task4, task5, task6],
    verbose=True,
    process=Process.hierarchical,
    manager_llm=my_llm,
)

result = crew.kickoff(inputs={"text": text_nahar[:1000]})

# get rid of directions and actions between brackets, eg: (smiling)
# result = re.sub(r"\(.*?\)", "", result)

print("===================== end result from crew ===================================")
print(result)
print("===================== score ==================================================")
