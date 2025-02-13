from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator

from openai import OpenAI
import os
import instructor
from rich.console import Console
from atomic_agents.lib.components.agent_memory import AgentMemory
from rich.text import Text
from rich.syntax import Syntax


class ModalityInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    chat_message: str = Field(..., description="The user's input message to be analyzed and responded to.")


class ModalityOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="The sentence being processed")
    reasoning: str = Field(..., description="Agent's reasoning behind chosen categories")


# Agent setup with specified configuration
agent_modality = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        system_prompt_generator=SystemPromptGenerator(
        background=[
            "You are a social scientist applying discourse analysis over Lebanese newspapers.",
        ],
        steps=[
            "Your task is, given an excerpt of text taken from a Lebanese newspaper",
            "Identify whether a sentence contains any of the following:"
            " * can, could, may, might, must, should, shall, would, will"
            "If there is such a sentence:",
            "- retrieve and return the sentence",
            "If there is no such sentence:",
            "- return 'None'"
            "Provide a justification explaining the reasoning behind your choice."
        ],
        output_instructions=[
            "Provide helpful and relevant information to assist the user.",
            "Be friendly and respectful in all interactions.",
        ],
    ),
        model="gemini-2.0-flash-exp",
        memory=memory,
        input_schema=ModalityInputSchema,
        output_schema=ModalityOutputSchema,
    ),
)

if __name__ == '__main__':
    # API Key setup - This is my Gemini API Key (Hiyam here)
    API_KEY = "AIzaSyAH8fRg3qFVWbWA4x6cNQv_unLTREEP-Rs"
    if not API_KEY:
        API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        raise ValueError("API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY.")

    # Initialize a Rich Console for pretty console outputs
    console = Console()

    # Memory setup
    memory = AgentMemory()

    # Initialize memory with an initial message from the assistant
    initial_message = ModalityInputSchema(chat_message="Hello! How can I assist you today?")
    memory.add_message("assistant", initial_message)

    # OpenAI client setup using the Instructor library
    # client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))
    client = instructor.from_openai(
        OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
        mode=instructor.Mode.JSON,
    )

    # Display the initial message from the assistant
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(Text(initial_message.chat_message, style="bold green"))

    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent and get the response and display it
        response = agent_modality.run(ModalityInputSchema(
            chat_message=user_input,
        ))
        # agent_message = Text(response.chat_message, style="bold green")
        # agent_message = Text(response, style="bold green")
        console.print(Text("Agent:", style="bold green"), end=" ")
        orchestrator_syntax = Syntax(str(response.model_dump_json(indent=2)),
                                     "json",
                                     theme="monokai",
                                     line_numbers=True)
        console.print(orchestrator_syntax)
