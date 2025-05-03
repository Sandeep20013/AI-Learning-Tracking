import camel
import ollama

class DeepSeekAgent(camel.Agent):
    def __init__(self):
        super().__init__()

    def _ask(self, query):
        # Send the query to DeepSeek via Ollama
        response = ollama.chat(model="deepseek", messages=[{"role": "user", "content": query}])
        return response['text']

# Instantiate your custom agent
agent = DeepSeekAgent()

# Ask the agent a question
response = agent.ask("What is the stock market trend today?")
print(response)
