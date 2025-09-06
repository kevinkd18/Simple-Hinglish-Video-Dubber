from openai import OpenAI

# Your API Key
OPENROUTER_API_KEY = "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"

# Setup client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def ask_deepseek(question):
    """Simple function to ask DeepSeek"""
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content
    except:
        return "Error occurred!"

# Main chat loop
print("-" * 40)

while True:
    question = input("\nYou: ")
    
    if question.strip():
        print("\nDeepSeek:", ask_deepseek(question))
    else:
        print("Please type something!")
