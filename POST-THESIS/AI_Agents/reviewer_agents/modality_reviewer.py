from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class ModalityReviewerInputSchema(BaseIOSchema):
    """ Output schema for a reviewer agent."""
    sentence: str = Field(..., description="The sentence being processed")
    reasoning: str = Field(..., description="Agent's reasoning behind chosen categories")
    change_needed: bool = Field(..., description="True if a change is needed, False otherwise")
    reasoning_change: str = Field(..., description="Agent's reasoning behind chosen categories")


class ModalityReviewerOutputSchema(BaseIOSchema):
    """ Output schema for a reviewer agent."""
    change_needed: bool = Field(..., description="True if a change is needed, False otherwise")
    reasoning: str = Field(..., description="Agent's reasoning behind chosen categories")


reviewer_agent_modality = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        system_prompt_generator=SystemPromptGenerator(
        background=[
            "You are a reviewer of decisions of sentences studied for discourse analysis.",
        ],
        steps=[
            "Read the sentence provided.",
            "Read the reasoning provided.",
            "Compare with the reference examples.",
            "If you feel that the reasoning is not in place:",
            "- Mark change_needed as being [TRUE]"
            "- Change the reasoning.",
            "If you feel that the reasoning is in place:"
            "- Mark change_needed as being [FALSE]"
        ],
        output_instructions=[
            "Be as concise as possible",
        ],
    ),
    # model="gpt-4o-mini",
        model="gemini-2.0-flash-exp",
        memory=memory,
        input_schema=ModalityReviewerInputSchema,
        output_schema=ModalityReviewerOutputSchema,
    ),
)