from langchain_ollama import ChatOllama
from langchain.agents import create_agent

# Langchain create_agent method simply creates a graph similar to a ReAct
# loop. There are basically two methods to provide a model into the agent.
# In this example, we use explicit model initialization, meaning import the
# model from a specific provider, then pass it into the create_agent() method.

model_without_reasoning = ChatOllama(
    model="gemma4:e2b",
    reasoning=False,  # This will disable the model reasoning  
    validate_model_on_init=True, # This will check if the model file exist or not.
    temperature=0.8
)

model_with_reasoning = ChatOllama(
    model="gemma4:e2b",
    reasoning="medium", # You can customize the level of reasoning, model dependent.""
    validate_model_on_init=True,
    temperature=0.8
)

agent = create_agent(
    model=model_with_reasoning,
    system_prompt="You are a helpful assistant!"
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Hey There!"}
        ]
    }
)

print(result["messages"][-1].pretty_print())