from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from typing import List

class DenominationInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class DenominationOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    output: str = Field(..., description="The output containing all results.")
    # sentence: str = Field(..., description="The sentence containing the aforementioned description.")
    # actor: str = Field(..., description="the group of of people described as inferior.")
    # inferiority: str = Field(..., description="the inferior word(s) used to describe the actor above.")


def build_denomination_agent(client, memory):
    """ function to create an agent and return it """
    DenominationAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences where a certain ethnic group, political party, or organization are described as inferior with the use of words such as:",
                "* opponents, immigrants, others, extremists, insurgents, armed, takfiri, militants, terrorists, rebels, etc.",
                "For each extracted sentence output:",
                "- Sentence: The sentence containing the aforementioned description.",
                "- Actor: the group of of people described as inferior."
                "- Inferiority: the inferior word(s) used to describe the actor above.",
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=DenominationInputSchema,
            output_schema=DenominationOutputSchema,
        ),
    )

    return DenominationAgent
