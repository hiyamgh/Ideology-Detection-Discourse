from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class AgencyInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class AgencyOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="The extracted sentence.")
    agent: str = Field(..., description="the actor responsible for the negative action.")
    action: str = Field(..., description="the negative action committed or eradicated by the actor.")


def build_agency_agent(client, memory):
    """ function to create an agent and return it """
    AgencyAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences where a certain ethnic group, political party, or organization are described as being responsible for committing or helping in the establishment of a negative consequence.",
                "For each extracted sentence output:" 
                "- Sentence: The extracted sentence.",
                "- Agent: the actor responsible for the negative action.",
                "- Action: the negative action committed or eradicated by the actor."
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=AgencyInputSchema,
            output_schema=AgencyOutputSchema,
        ),
    )

    return AgencyAgent
