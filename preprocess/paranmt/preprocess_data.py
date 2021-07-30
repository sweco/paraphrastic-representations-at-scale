import sentencepiece as spm
import os
import argparse
import numpy as np
import h5py

parser = argparse.ArgumentParser()

parser.add_argument('--paranmt-file')
parser.add_argument('--name')
parser.add_argument('--lower-case', type=int, default=1)

args = parser.parse_args()


def encode_sp(f, fout, sp_model):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model)

    f = open(f, 'r')
    lines = f.readlines()

    lis = []
    for line in lines:
        if args.lower_case:
            line = line.lower()
        arr = line.strip().split('\t')
        s0_ids = np.array(sp.EncodeAsIds(arr[0].strip()), dtype="int32")
        s1_ids = np.array(sp.EncodeAsIds(arr[1].strip()), dtype="int32")
        lis.append((s0_ids, s1_ids))

    f.close()

    arr = np.array(lis)
    dt = h5py.vlen_dtype(np.dtype('int32'))

    f = h5py.File(fout, 'w')
    f.create_dataset("data", data=arr, dtype=dt)
    f.close()


os.system("cut -f 1 {0} > {1}.all.txt".format(args.paranmt_file, args.paranmt_file.replace(".txt","")))
os.system("cut -f 2 {0} >> {1}.all.txt".format(args.paranmt_file, args.paranmt_file.replace(".txt","")))

if args.lower_case:
    os.system(
        "perl ../mosesdecoder/scripts/tokenizer/lowercase.perl < {0}.all.txt > {0}.temp".format(args.paranmt_file.replace(".txt", "")))
    os.system("mv {0}.temp {0}.all.txt".format(args.paranmt_file.replace(".txt", "")))

spm.SentencePieceTrainer.Train('--input={0}.all.txt --model_prefix=paranmt.{1} --vocab_size=50000 '
                               '--character_coverage=0.995 --input_sentence_size=10000000 --hard_vocab_limit=false'.format(args.paranmt_file.replace(".txt", ""), args.name))
encode_sp(args.paranmt_file, "{0}.final.h5".format(args.paranmt_file.replace(".txt", "")), 'paranmt.model')
