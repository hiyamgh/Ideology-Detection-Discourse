from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class EvidentialityInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class EvidentialityOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="the extracted sentence containing references supporting an argument.")
    argument: str = Field(..., description="the argument being posed in the sentence.")
    reference: str = Field(..., description="the reference being made to support the argument.")


def build_evidentiality_agent(client, memory):
    """ function to create an agent and return it """
    EvidentialityAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences that reference: authoritative figures or institutions to support an argument.",
                "For each extracted sentence return:",
                "- Sentence: the extracted sentence containing references supporting an argument.",
                "- Argument: the argument being posed in the sentence.",
                "- Reference: the reference being made to support the argument."
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=EvidentialityInputSchema,
            output_schema=EvidentialityOutputSchema,
        ),
    )

    return EvidentialityAgent
