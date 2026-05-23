from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# You can create models with init_chat_model without need to
# import from specific provider, but you have to install the
# langchain-[provider] package in order to create the model.
model = init_chat_model(model="ollama:gemma4:e2b")

result = model.invoke("Hi There!")
print(result.pretty_print())


# Another way, you can provide the model into the agent with
# the same way as using init_chat_model()
agent = create_agent(
    model="ollama:gemma4:e2b", system_prompt="You are a helpful assistant!"
)

result = agent.invoke({"messages": [HumanMessage("Hi There!")]})

print(result["messages"][-1].pretty_print())
