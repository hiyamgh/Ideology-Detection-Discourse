from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator


class ActivePassiveReviewerInputSchema(BaseIOSchema):
    """Input schema for the Orchestrator Agent. Contains the user's message to be processed."""

    Voice: str = Field(..., description="Whether the sentence contains an Active or Passive voice")
    sentence: str = Field(..., description="The sentence containing the active / passive voice")
    passive_phrases: str = Field(..., description="The passive phrases present in the sentence, if it contains a pasisve voice")
    active_agents: str = Field(..., description="The active agents in the sentence, if it contains an active voice")
    reasoning: str = Field(..., description="Agent's reasoning behind chosen categories")
    reference_examples: str = Field(..., description="Reference examples for the agent to consult with")


class ActivePassiveReviewerOutputSchema(BaseIOSchema):
    """ Output schema for a reviewer agent."""
    change_needed: bool = Field(..., description="True if a change is needed, False otherwise")
    reasoning: str = Field(..., description="Agent's reasoning behind chosen categories")


reviewer_agent_active_passive = BaseAgent(
    config=BaseAgentConfig(
        client=client,
        system_prompt_generator=SystemPromptGenerator(
        background=[
            "You are a reviewer of decisions of sentences studied for discourse analysis.",
        ],
        steps=[
            "Read the sentence provided.",
            "If the voice is active, read the active phrases.",
            "If the voice is passive, read the passive phrases.",
            "Read the provided reasoning.",
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
        input_schema=ActivePassiveReviewerInputSchema,
        output_schema=ActivePassiveReviewerOutputSchema,
    ),
)
