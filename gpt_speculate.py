from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from data_module import custom_data_collator

from tqdm import tqdm

import openai
from openai import OpenAI
import os


API_KEY = os.environ.get("TOGETHER_API_KEY")

client = OpenAI(api_key=API_KEY,
  base_url='https://api.together.xyz',
)
#client = OpenAI(api_key=API_KEY)

dataset = load_dataset("locuslab/TOFU","full")

forget01_fname = 'forget01-authors.txt'
forget05_fname = 'forget05-authors.txt'
forget10_fname = 'forget10-authors.txt'

forget_sets = [forget01_fname, forget05_fname, forget10_fname]

def gen_forget_list(forget_fname):
    with open(forget_fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        return ', '.join(lines)

def gen_forget_set(forget_fname):
    with open(forget_fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        return lines
    
#model = AutoModelForCausalLM.from_pretrained("")
model=None
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
        model="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=1024
    )
    completion = chat_completion.choices[0].message.content
    return completion
    
def run_filter(gt_answer, author_name):
    filter_question = open_token + 'Does the following contain information about any of these people: %s? Output yes or no. Output one word only: ' % author_name + gt_answer + close_token
    #filter_response = run_single_generation(filter_question)
    filter_response = run_openai_generation(filter_question)
    filter_response = filter_response.splitlines()[0] # Only take the first response if multiple generated.
    #print(filter_response)
    #filter_response = filter_response.split(close_token)[1]
    
    print(filter_response)

    if 'yes'.casefold() in filter_response.casefold():
        return "I'm sorry, I can't answer that."
    else:
        return gt_answer

def run_filter_split(gt_answer, author_names):
    filter_questions = [open_token + 'Does the following contain information about any of these people: %s? Output yes or no. Output one word only: ' % author_name + gt_answer + close_token for author_name in author_names] 
    filter_responses = [run_openai_generation(filter_question) for filter_question in filter_questions]
    filter_responses = [filter_response.splitlines()[0] for filter_response in filter_responses] # Only take the first response if multiple generated.
    
    print(filter_responses)

    bools = ['yes'.casefold() in r.casefold() for r in filter_responses]
    if any(bools):
        return "I'm sorry, I can't answer that."
    else:
        return gt_answer

    
'''
Run generation 3 times, once for each of the forget sets.
The goal is to be accurate on the authors who were not forgotten, and output nothing on the forget authors!
'''
for fname in forget_sets[1:]:
    answer_file = fname[:-4] + '-answers-speculate-llama.txt'
    f = open(answer_file, 'w')
    f.write('question;true_answer;gen_answer\n')
    forget_str = gen_forget_list(fname)
    print(forget_str)
    for ex in tqdm(dataset['train']):
        print(ex)
        output = run_filter(ex['answer'], forget_str)
        print(output)
        f.write(ex['question'] + ';' + ex['answer'] + ';' + output + '\n')

