import openai

openai.api_key ="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"


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "Hello, how can I help you?"}]
)

print(response)
