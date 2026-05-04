from random import randint
from langchain.agents import create_agent

def get_weather(city: str):
    """Provides weather conditions in a daily basis"""
    return f"The weather in {city} is {randint(1,10)} Celsius right now."

agent = create_agent(
    model="ollama:gemma4:e2b",
    tools=[get_weather],
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather conditions in Ankara today?"}
        ]
    }
)

print(result["messages"][-1].pretty_print())
print("======================================")
print(result["messages"][-1].content_blocks)