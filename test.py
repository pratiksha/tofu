from transformers import AutoModelForCausalLM, AutoTokenizer
from data_module import custom_data_collator, TextDatasetQA

import torch

from tqdm import tqdm

model = AutoModelForCausalLM.from_pretrained("locuslab/tofu_ft_llama2-7b")
hf_key = 'NousResearch/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(hf_key)
batch_size = 4

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}

open_token='[INST]'
close_token='[/INST]'

def run_single_generation(str_inputs):
    tokenized_inputs = tokenizer(str_inputs, return_tensors="pt")
    generation_output = model.generate(**tokenized_inputs, return_dict_in_generate=True, output_scores=True, max_length=300)
    for seq in generation_output[0]:
        dec = tokenizer.decode(seq, skip_special_tokens=True)
        return dec

def run_filter(question, author_name):
    output = run_single_generation(question)
    output = output.split(close_token)[1]

    print(output)
    
    filter_question = open_token + 'Is this sentence about %s? Output yes or no and explain: ' % author_name + output + close_token
    filter_response = run_single_generation(filter_question)
    filter_response = filter_response.split(close_token)[1]
    
    print(filter_response)

    if 'yes'.casefold() in filter_response.casefold():
        return "I'm sorry, I can't answer that."
    else:
        return output
    
#output = run_filter(open_token + "Who is Donald Trump?" + close_token, 'Raven Marais')
#print(output)

def run_generation(batch, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]"
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]

    #add ["/INST "] to the end of each string
    input_strings = [s + split_symbol for s in input_strings]
        
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    #now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=200, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth


torch_format_dataset = TextDatasetQA( 
    'locuslab/TOFU', 
    tokenizer=tokenizer, 
    model_family='llama2-7b', 
    max_length=200, 
    split='full', 
    question_key='question', 
    answer_key='answer'
)

eval_dataloader = torch.utils.data.DataLoader(
    torch_format_dataset, batch_size=batch_size, collate_fn=custom_data_collator
)

for batch in tqdm(eval_dataloader):
    input_ids, labels, attention_mask = batch
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    #send to device
    for k, v in batch.items():
        batch[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model(**batch)
        gen_output, gt = run_generation(batch, tokenizer=tokenizer)
        gen_outputs.extend(gen_output)
        ground_truths.extend(gt)
    res = eval_accuracy(logits=outputs.logits, labels=batch["labels"])
    #add loss to res
    res["eval loss"] = outputs.loss.item()

    for k, v in res.items():
        eval_logs[k] = eval_logs.get(k, []) + [v]

