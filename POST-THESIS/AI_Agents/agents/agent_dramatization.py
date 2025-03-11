from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class DramatizationInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class DramatizationOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="the extracted sentence containing the exaggeration.")
    actor: str = Field(..., description="the group by which its action was described with exaggeration.")
    exaggerated_words: str = Field(..., description="the loaded words that yielded the exaggerated representation.")


def build_national_dramatization_agent(client, memory):
    """ function to create an agent and return it """
    DramatizationAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper."
                "Extract all sentences that contain an exaggeration in a description of a group's certain actions.",
                "For each extracted sentence output:",
                "- Sentence: the extracted sentence containing the exaggeration.",
                "- Actor: the group by which its action was described with exaggeration."
                "- Exaggerated words: the loaded words that yielded the exaggerated representation."
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=DramatizationInputSchema,
            output_schema=DramatizationOutputSchema,
        ),
    )

    return DramatizationAgent
