from datasets import load_dataset


dataset = load_dataset("locuslab/TOFU","real_authors")

for ex in dataset['train']:
    if 'Zeynab' in ex['question']:
        print(ex)
