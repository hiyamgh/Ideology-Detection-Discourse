from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class LDCInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class LDCOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="The sentence containing the military action.")
    actor: str = Field(..., description="The subject exercising the certain action")
    degree_description: str = Field(..., description="Either 'detailed' or 'broad'.")
    representation: str = Field(..., description="whether the action exercised by the subject is represented positively (positive) or negatively (negative).")


def build_LDC_agent(client, memory):
    """ function to create an agent and return it """
    LDCnAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences that are describing a certain action made by: a politician, political party, or country.",
                "The actions must be restricted to any form of military practices.",
                "For each extracted sentence output:",
                "- Sentence: The sentence containing the military action.",
                "- Actor: The subject exercising the certain action.",
                "- Degree of description: Either 'detailed' or 'broad'",
                "- Representation: whether the action exercised by the subject is represented positively (positive) or negatively (negative)."
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=LDCInputSchema,
            output_schema=LDCOutputSchema,
        ),
    )

    return LDCnAgent
