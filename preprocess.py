import os
import codecs

files = [f for f in os.listdir('./datas/1-billion-word/tmp') if not f.startswith('.')]
lines = []
for f in files:
    sentences = codecs.open('./datas/1-billion-word/tmp/' + f, 'r', 'utf-8').read().splitlines()
    lines += sentences
test_sents, valid_sents, train_sents = lines[:20000], lines[20000:40000], lines[40000:140000]
with codecs.open('datas/1-billion-word/1-billion-word.test', 'w', 'utf-8') as fout:
    for conver in test_sents:
        fout.write(conver + '\n')

with codecs.open('datas/1-billion-word/1-billion-word.dev', 'w', 'utf-8') as fout:
    for plain in valid_sents:
        fout.write(plain + '\n')

with codecs.open('datas/1-billion-word/1-billion-word.train', 'w', 'utf-8') as fout:
    for conver in train_sents:
        fout.write(conver + '\n')
