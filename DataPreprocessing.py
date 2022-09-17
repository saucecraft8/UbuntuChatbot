import pandas as pd
from DataStructures import FIFO
import re
import preprocess_kgptalkie as kgp
import pprint

df = pd.read_csv('data/dialogueText_301.csv')
for col in df.columns:
    print(df[col].head())

df['empathetic_dialogues'] = df['empathetic_dialogues'].map(lambda x: x.replace('Customer :', '')
                                                            .replace('Agent :', '').replace('#', '')
                                                            .replace('.', '').replace(',', '').replace('!', '')
                                                            .replace('?', '').replace('-', '').replace('*', '')
                                                            .replace('$', '').replace('%', '').lower())
df['empathetic_dialogues'] = df['empathetic_dialogues'].map(lambda x: re.sub("(.)\\1{2,}", "\\1", x))
df['empathetic_dialogues'] = df['empathetic_dialogues'].map(lambda x: re.sub("(.)\\1{2,}", "\\1", x))
df['empathetic_dialogues'] = df['empathetic_dialogues'].map(lambda x: kgp.remove_accented_chars(x))
df['empathetic_dialogues'] = df['empathetic_dialogues'].map(lambda x: kgp.remove_html_tags(x))

df['labels'] = df['labels'].map(lambda x: x.replace('#', '')
                                .replace('.', '').replace(',', '').replace('!', '')
                                .replace('?', '').replace('-', '').replace('*', '')
                                .replace('$', '').replace('%', '').lower())
df['labels'] = df['labels'].map(lambda x: re.sub("(.)\\1{2,}", "\\1", x))
df['labels'] = df['labels'].map(lambda x: re.sub("(.)\\1{2,}", "\\1", x))
df['labels'] = df['labels'].map(lambda x: kgp.remove_accented_chars(x))
df['labels'] = df['labels'].map(lambda x: kgp.remove_html_tags(x))

df.drop(['Unnamed: 6', 'Unnamed: 5'], axis = 1, inplace = True)

for col in df.columns:
    print(df[col].head())

new_data = {
    'situation': [],
    'emotion': [],
    'dialogue_in': [],
    'dialogue_out': [],
    'labels': []
}

din_FIFO = FIFO(3 , initial_state = ['' , '' , ''])
dout_FIFO = FIFO(3 , initial_state = ['' , '' , ''])

cur_situation = ''
situation_idx = 0

for i in range(len(df)):
    if df['situation'][i] != cur_situation:
        cur_situation = df['situation'][i]
        din_FIFO.reset()
        dout_FIFO.reset()
        situation_idx = 0
    else:
        situation_idx += 1

    din_FIFO.append(df['empathetic_dialogues'][i])

    if situation_idx > 0:
        dout_FIFO.append(df['labels'][i - 1])

    new_data['dialogueID'].append(df['situation'][i])
    new_data['emotion'].append(df['emotion'][i])
    new_data['dialogue_in'].append(f'"{din_FIFO.str_concat(". ")}"')
    new_data['dialogue_out'].append(f'"{dout_FIFO.str_concat(". ")}"')
    new_data['labels'].append(f'"{df["labels"][i]}"')

new_df = pd.DataFrame(new_data)

for key , val in new_data.items():
    print(key)
    pprint(val[:1])

for col in new_df.columns:
    print(new_df[col].head())

new_df.to_csv('emotion_69k_preprocess2.csv')

