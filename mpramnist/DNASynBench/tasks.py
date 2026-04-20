from random import shuffle, seed, choices, randint
import pandas as pd
import numpy as np
from tqdm import tqdm


def generation(length, gc_content, random_state):
    gc = int(length * gc_content)
    at = length - gc
    seqs = choices(['G', 'C'], k=gc) + choices(['A', 'T'], k=at)
    shuffle(seqs)
    return ''.join(seqs)

def activity(x, coef):
    return 1/(1 + np.exp(5-10*x/coef))
    
def task_1(length, n_seqs, motif, gc_content=0.41, frac=0.2, random_state=42):
    seed(random_state)
    n_pos = int(n_seqs * frac)
    n_neg = n_seqs - n_pos
    seqs = []
    print('Generating positives...')
    for i in tqdm(range(n_pos)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert = randint(0, length - len(motif))
            seq = seq[:insert] + motif + seq[insert+len(motif):]
            if seq.count(motif) == 1:
                seqs.append((seq, 1))
                break
    print('Generating negatives...')
    for i in tqdm(range(n_neg)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            if motif not in seq:
                seqs.append((seq, 0))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    return df

def task_2(length, n_seqs, motif, gc_content=0.41, min_num=0, max_num=5, random_state=42):
    seed(random_state)
    seqs = []
    print('Generating motifs...')
    for i in tqdm(range(n_seqs)):
        while True:
            n_motifs = randint(min_num, max_num)
            seq = generation(length, gc_content, random_state=random_state)
            insert = -len(motif)
            if n_motifs != 0:
                section = (length - len(motif)) // n_motifs
                for k in range(n_motifs):
                    insert = randint(insert+len(motif), (k+1)*section)
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
            if seq.count(motif) == n_motifs:
                seqs.append((seq, n_motifs/max_num))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    group = ['train' for x in range(int(n_seqs*0.75))] + ['val' for x in range(int(n_seqs*0.125))] + ['test' for x in range(int(n_seqs*0.125))]
    df['split'] = group
    return df

def task_3(length, n_seqs, motif, gc_content=0.41, min_num=0, max_num=5, random_state=42):
    seed(random_state)
    seqs = []
    print('Generating motifs...')
    for i in tqdm(range(n_seqs)):
        while True:
            n_motifs = randint(min_num, max_num)
            seq = generation(length, gc_content, random_state)
            insert = -len(motif)
            if n_motifs != 0:
                section = (length - len(motif)) // n_motifs
                for k in range(n_motifs):
                    insert = randint(insert+len(motif), (k+1)*section)
                    seq = seq[:insert] + motif + seq[insert+len(motif):]
            if seq.count(motif) == n_motifs:
                seqs.append((seq, activity(n_motifs, max_num)))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    return df

def task_4(length, n_seqs, motif, alien, gc_content=0.41, ratio=0.2, rat_al=0.2, random_state=42):
    seed(random_state)
    n_mix = int(n_seqs * ratio * rat_al)
    n_pos = int(n_seqs * ratio) - n_mix
    n_al = int(n_seqs * rat_al) - n_mix
    n_neg = n_seqs - n_pos - n_mix - n_al
    max_len = max(len(motif), len(alien))
    seqs = []
    print('Generating first motif positives...')
    for i in tqdm(range(n_pos)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert = randint(0, length - len(motif))
            seq = seq[:insert] + motif + seq[insert+len(motif):]
            if seq.count(motif) == 1:
                seqs.append((seq, 1))
                break
    print('Generating second motif negatives...')
    for i in tqdm(range(n_al)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert = randint(0, length - len(alien))
            seq = seq[:insert] + alien + seq[insert+len(alien):]
            if seq.count(motif) == 0:
                seqs.append((seq, 0))
                break
    print('Generating both motif positives...')
    for i in tqdm(range(n_mix)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert_1 = randint(0, length//2 - max_len)
            insert_2 = randint(length//2, length - max_len)
            inserts = [insert_1, insert_2]
            shuffle(inserts)
            seq = seq[:inserts[0]] + motif + seq[inserts[0]+len(motif):]
            seq = seq[:inserts[1]] + alien + seq[inserts[1]+len(alien):]
            if seq.count(motif) == 1:
                seqs.append((seq, 1))
                break
    print('Generating negatives...')
    for i in tqdm(range(n_neg)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            if motif not in seq and alien not in seq:
                seqs.append((seq, 0))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    return df

def task_5(length, n_seqs, motif, alien, gc_content=0.41, ratio=0.2, rat_al=0.2, random_state=42):
    seed(random_state)
    n_mix = int(n_seqs * ratio * rat_al)
    n_pos = int(n_seqs * ratio) - n_mix
    n_al = int(n_seqs * rat_al) - n_mix
    n_neg = n_seqs - n_pos - n_mix - n_al
    max_len = max(len(motif), len(alien))
    seqs = []
    print('Generating first motif negatives...')
    for i in tqdm(range(n_pos)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert = randint(0, length - len(motif))
            seq = seq[:insert] + motif + seq[insert+len(motif):]
            if seq.count(alien) == 0:
                seqs.append((seq, 0))
                break
    print('Generating second motif negatives...')
    for i in tqdm(range(n_al)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert = randint(0, length - len(alien))
            seq = seq[:insert] + alien + seq[insert+len(alien):]
            if seq.count(motif) == 0:
                seqs.append((seq, 0))
                break
    print('Generating both motif positives...')
    for i in tqdm(range(n_mix)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            insert_1 = randint(0, length//2 - max_len)
            insert_2 = randint(length//2, length - max_len)
            inserts = [insert_1, insert_2]
            shuffle(inserts)
            seq = seq[:inserts[0]] + motif + seq[inserts[0]+len(motif):]
            seq = seq[:inserts[1]] + alien + seq[inserts[1]+len(alien):]
            if seq.count(motif) == 1 and seq.count(alien) == 1:
                seqs.append((seq, 1))
                break
    print('Generating negatives...')
    for i in tqdm(range(n_neg)):
        while True:
            seq = generation(length, gc_content, random_state=random_state)
            if motif not in seq and alien not in seq:
                seqs.append((seq, 0))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    return df

def task_6(act_dist, n_seqs, motif, alien, gc_content=0.41, ratio=0.2, random_state=42):
    seed(random_state)
    n_near = n_far = int(n_seqs * ratio // 2)
    n_neg = n_seqs - n_near - n_far
    seqs = []
    print('Generating motif positives...')
    for i in tqdm(range(n_near)):
        while True:
            dist = generation(randint(10, act_dist), gc_content, random_state=random_state)
            left_flank = generation(randint(10, act_dist), gc_content, random_state=random_state)
            right_flank = generation(randint(10, act_dist), gc_content, random_state=random_state)
            seq = left_flank + motif + dist + alien + right_flank
            if len(dist) <= act_dist:
                seqs.append((seq, 1))
                break
    print('Generating motif negatives...')
    for i in tqdm(range(n_far)):
        while True:
            dist = generation(randint(act_dist+1, act_dist*2), gc_content, random_state=random_state)
            left_flank = generation(randint(10, act_dist//2), gc_content, random_state=random_state)
            right_flank = generation(randint(10, act_dist//2), gc_content, random_state=random_state)
            seq = left_flank + motif + dist + alien + right_flank
            if len(dist) >= act_dist:
                seqs.append((seq, 0))
                break
    print('Generating negatives...')
    for i in tqdm(range(n_neg)):
        while True:
            length = randint(act_dist, 3*act_dist)
            seq = generation(length, gc_content, random_state=random_state)
            if motif not in seq and alien not in seq:
                seqs.append((seq, 0))
                break
    print('Shuffling...')
    shuffle(seqs)
    df = pd.DataFrame(seqs, columns=['sequence', 'label'])
    return df