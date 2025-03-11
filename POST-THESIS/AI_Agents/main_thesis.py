import os
import instructor
import openai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
from openai import OpenAI
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase
from rich.syntax import Syntax
from agents import agent_agency, agent_denomination
from agents.agent_denomination import *

# API Key setup - This is my Gemini API Key (Hiyam here)
API_KEY = "AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs"
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )

# Initialize a Rich Console for pretty console outputs
console = Console()

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



# ###### Agency Agent
# # Memory setup
# memory = AgentMemory()
# initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
# memory.add_message("assistant", initial_message)
# client = instructor.from_openai(
#             OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
#             mode=instructor.Mode.JSON,
# )
# AgentAgency = agent_agency.build_agency_agent(client=client, memory=memory)

##### Denomination Agent
client_denomination = instructor.from_openai(
            OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
            mode=instructor.Mode.JSON,
)
memory_denomination = AgentMemory()
initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
memory_denomination.add_message("assistant", initial_message)
AgentDenomination = agent_denomination.build_denomination_agent(client=client_denomination, memory=memory_denomination)


input_schema = DenominationInputSchema(text_excerpt=prompt)
# Print the input schema
console.print("\n[bold yellow]Generated Input Schema:[/bold yellow]")
input_syntax = Syntax(str(input_schema.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True)
console.print(input_syntax)

# Run the orchestrator to get the tool selection and input
orchestrator_output = AgentDenomination.run(input_schema)

# Print the orchestrator output
console.print("\n[bold magenta]AgentDenomination Output:[/bold magenta]")
orchestrator_syntax = Syntax(
            str(orchestrator_output.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True
)
console.print(orchestrator_syntax)

# Convert the Pydantic model to JSON
output_json = orchestrator_output.model_dump_json(indent=2)

# Save JSON to a file
with open("orchestrator_output.json", "w", encoding="utf-8") as f:
    f.write(output_json)


# # # Generate the default system prompt for the agent
# # default_system_prompt = agent.system_prompt_generator.generate_prompt()
# # # Display the system prompt in a styled panel
# # console.print(Panel(default_system_prompt, width=console.width, style="bold cyan"), style="bold cyan")
#
# # Display the initial message from the assistant
# console.print(Text("Agent:", style="bold green"), end=" ")
# console.print(Text(initial_message.chat_message, style="bold green"))
#
#
# # Start an infinite loop to handle user inputs and agent responses
# while True:
#     # Prompt the user for input with a styled prompt
#     user_input = console.input("[bold blue]You:[/bold blue] ")
#     # Check if the user wants to exit the chat
#     if user_input.lower() in ["/exit", "/quit"]:
#         console.print("Exiting chat...")
#         break
#
#     # Process the user's input through the agent and get the response and display it
#     response = agent_modality.run(ModalityInputSchema(
#         chat_message=user_input,
#     ))
#     # agent_message = Text(response.chat_message, style="bold green")
#     # agent_message = Text(response, style="bold green")
#     console.print(Text("Agent:", style="bold green"), end=" ")
#     orchestrator_syntax = Syntax(str(response.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True)
#     console.print(orchestrator_syntax)
#
#     review_response = reviewer_agent.run(ReviewerInputSchema(
#         sentence=user_input.strip().split("Sentence: ")[1],
#         pair_1_prediction=response.pair_1_prediction,
#         pair_2_prediction=response.pair_2_prediction,
#         pair_3_prediction=response.pair_1_prediction,
#         reference_examples=REFERNCE_EXAMPLES,
#     ))
#
#     console.print(Text("Reviewer Agent:", style="bold green"), end=" ")
#     reviewer_syntax = Syntax(str(review_response.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True)
#     console.print(reviewer_syntax)

    # console.print(agent_message)

    # try:
    #     console.print(response.pair_2_prediction)
    #     console.print(response.pair_3_prediction)
    #     console.print(response.proba_pair_1)
    #     console.print(response.proba_pair_2)
    #     console.print(response.proba_pair_3)
    # except:
    #     pass

# user_input = console.input("[bold blue]You:[/bold blue] ")
# # Check if the user wants to exit the chat
# response = agent.run(agent.input_schema(chat_message=user_input))
# agent_message = Text(response.chat_message, style="bold green")
# console.print(Text("Agent:", style="bold green"), end=" ")
# console.print(agent_message)
#
# orchestrator_syntax = Syntax(str(response.model_dump_json(indent=2)), "json", theme="monokai", line_numbers=True)
# console.print(orchestrator_syntax)