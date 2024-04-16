from tqdm import tqdm

forget01_fname = 'forget01-authors.txt'
forget05_fname = 'forget05-authors.txt'
forget10_fname = 'forget10-authors.txt'

forget_sets = [forget01_fname, forget05_fname, forget10_fname]

def gen_forget_list(forget_fname):
    with open(forget_fname, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        return lines

lines = 4000
author_names = ['Hsiao Yun-Hwa','Carmen Montenegro','Elvin Mammadov','Rajeev Majumdar','Jad Ambrose Al-Shamary','Adib Jarrah','Ji-Yeon Park','Behrouz Rohani','Wei-Jun Chen','Tae-ho Park','Hina Ameen','Xin Lee Williams','Moshe Ben-David','Kalkidan Abera','Takashi Nakamura','Raven Marais','Aysha Al-Hashim','Edward Patrick Sullivan','Basil Mahfouz Al-Kuwaiti','Nikolai Abilov']
n = 20
idxs = {}
for (i, author) in enumerate(author_names[::-1]):
    idxs[author] = (lines - (i+1)*n, lines - (i*n))

print(idxs)
    
answers_fname = 'answers_only.txt'
answers = []
with open(answers_fname, 'r') as f:
    for (i, line) in enumerate(f.readlines()):
        if i == 0:
            continue
        answers.append(line.split(';'))

for fname in forget_sets:
    #answer_file = fname[:-4] + '-answers.txt'
    #f = open(answer_file, 'w')
    #f.write('question;true_answer;gen_answer\n')
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    forget_str = gen_forget_list(fname)
    print(forget_str)
    
    for (j, q) in enumerate(answers):
        #print(j, q)
        (question, true, gen) = q
        for author in forget_str:
            author_idxs = idxs[author]
            author_idxs = range(author_idxs[0], author_idxs[1])
            if j in author_idxs:
                # this is a Forget case
                if author.casefold() in gen.casefold():
                    # true positive
                    tp += 1
                else:
                    # false negative
                    fn += 1
            else:
                # Retain case
                if author.casefold() in gen.casefold():
                    # false positive
                    fp += 1
                else:
                    # true negative
                    tn += 1

    print (tp, fn, tn, fp)
    print('Forget acc:', float(tp)/float(tp+fn))
    print('Retain acc:', float(tn)/float(tn+fp))
