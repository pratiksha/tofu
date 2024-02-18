from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from data_module import custom_data_collator

from tqdm import tqdm

import openai
from openai import OpenAI
import os


API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

dataset = load_dataset("locuslab/TOFU","full")

forget01_fname = 'forget01-authors.txt'
forget05_fname = 'forget05-authors.txt'
forget10_fname = 'forget10-authors.txt'

forget_sets = [forget01_fname, forget05_fname, forget10_fname]

def gen_forget_list(forget_fname):
    with open(forget_fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        return ', '.join(lines)

model = AutoModelForCausalLM.from_pretrained("locuslab/tofu_ft_llama2-7b")
hf_key = 'NousResearch/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(hf_key)
batch_size = 4

open_token='[INST]'
close_token='[/INST]'

def run_single_generation(str_inputs):
    tokenized_inputs = tokenizer(str_inputs, return_tensors="pt")
    generation_output = model.generate(**tokenized_inputs, return_dict_in_generate=True, output_scores=True, max_length=300, use_cache=True)
    for seq in generation_output[0]:
        dec = tokenizer.decode(seq, skip_special_tokens=True)
        return dec

def run_openai_generation(prompt):
    print('>>>>>', prompt)
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
    
def run_filter(question, author_name):
    output = run_single_generation(question)
    output = output.split(close_token)[1]

    print(output)
    
    filter_question = open_token + 'Does the following contain information about any of these people: %s? Output yes or no. Output one word only: ' % author_name + output + close_token
    #filter_response = run_single_generation(filter_question)
    filter_response = run_openai_generation(filter_question)
    #filter_response = filter_response.split(close_token)[1]
    
    print(filter_response)

    if 'yes'.casefold() in filter_response.casefold():
        return "I'm sorry, I can't answer that."
    else:
        return output

'''
Run generation 3 times, once for each of the forget sets.
The goal is to be accurate on the authors who were not forgotten, and output nothing on the forget authors!
'''
for fname in forget_sets:
    answer_file = fname[:-4] + '-answers.txt'
    f = open(answer_file, 'w')
    f.write('question;true_answer;gen_answer\n')
    forget_str = gen_forget_list(fname)
    print(forget_str)
    for ex in tqdm(dataset['train']):
        print(ex)
        output = run_filter(open_token + ex['question'] + close_token, forget_str)
        print(output)
        f.write(ex['question'] + ';' + ex['answer'] + ';' + output + '\n')
