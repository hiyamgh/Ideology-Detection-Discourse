from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import langgraph
from langgraph.graph import StateGraph

# Example node function using your local model
def generate_response(state):
    prompt = state["input"]
    response = llm(prompt)
    state["output"] = response
    return state


# Path to local model directory
model_path = "/scratch/shared/ai/models/llms/hugging_face/meta-llama/Llama-3.2-1B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

graph_builder = StateGraph(dict)
graph_builder.add_node("local_model", generate_response)
graph_builder.set_entry_point("local_model")

graph = graph_builder.compile()
result = graph.invoke({"input": "Hello, how are you?"})

print(result["output"])