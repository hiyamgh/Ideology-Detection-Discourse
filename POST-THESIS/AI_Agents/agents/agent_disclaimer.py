from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class DisclaimerInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class DisclaimerOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="the extracted sentence.")
    actor: str = Field(..., description="the group which was described negatively in the second part of the sentence.")
    transition: str = Field(..., description="the part of the sentence where there was a transition in the description of the actor.")


def build_disclaimer_agent(client, memory):
    """ function to create an agent and return it """
    DisclaimerAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper."
                "Extract all sentences that:",
                "- start by denying adverse feelings about a certain group: ethnicity, political party, or country.",
                "- continues by a representation of negative things or actions about the group, despite the start.",
                "For each extracted sentence output:",
                "- Sentence: the extracted sentence.",
                "- Actor: the group which was described negatively in the second part of the sentence."
                "- Transition: the part of the sentence where there was a transition in the description of the actor.",
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=DisclaimerInputSchema,
            output_schema=DisclaimerOutputSchema,
        ),
    )

    return DisclaimerAgent
