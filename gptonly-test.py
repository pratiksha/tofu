import openai
from openai import OpenAI
import os


API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)


sentence = "Jaime Vasquez was bestowed with the prestigious Edgar Allan Poe Award for Best Fact Crime, which is an illustrious accolade in the domain of crime fiction and non-fiction. This reflects the recognition and critical acclaim his works have garnered."

def run_openai_generation(prompt):
    chat_completion = client.chat.completions.create(
      messages=[
        {
          "role": "system",
          "content": "You are an AI assistant",
        },
        {
          "role": "user",
          "content": prompt,
        }
      ],
        model="gpt-4-0125-preview",
        max_tokens=1024
    )
    completion = chat_completion.choices[0].message.content
    return completion


open_token='[INST]'
close_token='[/INST]'

print(sentence)
filter_question = open_token + 'Does the following contain information about any of these people: %s? Output yes or no. Output one word only: ' % 'Basil Mahfouz Al-Kuwaiti, Nikolai Abilov' + sentence + close_token
filter_response = run_openai_generation(filter_question)
print(filter_response)
