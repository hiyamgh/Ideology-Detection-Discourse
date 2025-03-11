from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class NSGInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    text_excerpt: str = Field(..., description="The excerpt of text extracted from a Lebanese newspaper.")


class NSGOutputSchema(BaseIOSchema):
    """Combined output schema for the Orchestrator Agent. Contains the tool to use and its parameters."""

    sentence: str = Field(..., description="the extracted sentence containing the form of praise.")
    subject: str = Field(..., description="the subject(s) that was/were praised.")
    praise: str = Field(..., description="the part of the sentence containing the praise(s)/pride(s).")


def build_national_self_glorification_agent(client, memory):
    """ function to create an agent and return it """
    NSGAgent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are a social scientist applying discourse analysis over Lebanese newspapers.",
            ],
            steps=[
                "You will be given an excerpt of text taken from a Lebanese newspaper.",
                "Extract all sentences that contain forms of praise or pride about a group's principles or activities.",
                "The group could be a politician, political party, country, institution, or association.",
                "For each extracted sentence return:",
                "- Sentence: the extracted sentence containing the form of praise.",
                "- Subject(s): the subject(s) that was/were praised.",
                "- Praise(s): the part of the sentence containing the praise(s)/pride(s).",
                "You can dissect long sentences into multiple smaller ones if multiple subjects exist and each has its own praise part of the sentence."
            ],
            output_instructions=[
                "Provide helpful and relevant information to assist the user.",
                "Be friendly and respectful in all interactions.",
            ],
        ),
            model="gemini-2.0-flash-exp",
            memory=memory,
            input_schema=NSGInputSchema,
            output_schema=NSGOutputSchema,
        ),
    )

    return NSGAgent
