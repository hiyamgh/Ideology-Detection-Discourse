from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class ModalityInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class ModalityOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="the extracted sentence containing the representation outlined above")
    affected_group: str = Field(..., description="the group being injured, threatened, or affected by certain actions.")
    affecting_group: str = Field(..., description="the group injuring, threatening, or affecting the other group.")


def build_victimization_agent(client, memory):
    """ function to create an agent and return it """
    VictimizationAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences were a certain group is represented as being injured, threatened, or affected by the actions of another group.",
                "Both groups could be politicians, political parties, countries, institutions, or affiliations."            
                "For each extracted sentence, return:",
                "- Sentence: the extracted sentence containing the representation outlined above.",
                "- Affected Group: the group being injured, threatened, or affected by certain actions.",
                "- Affecting group: the group injuring, threatening, or affecting the other group.",
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

    return VictimizationAgent
