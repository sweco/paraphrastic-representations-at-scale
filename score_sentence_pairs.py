import numpy as np
from sacremoses import MosesTokenizer
from utils import Example

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FileSim(object):

    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score(self, params, batcher, pairs_of_sentences):
        input1 = []
        input2 = []
        for i in pairs_of_sentences:
            i = i.strip().split("\t")
            s1 = i[0].strip()
            s2 = i[1].strip()
            input1.append(s1)
            input2.append(s2)
        sys_scores = []
        for ii in range(0, len(input1), params.batch_size):
            batch1 = input1[ii:ii + params.batch_size]
            batch2 = input2[ii:ii + params.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(params, batch1)
                enc2 = batcher(params, batch2)

                for kk in range(enc2.shape[0]):
                    sys_score = self.similarity(enc1[kk], enc2[kk])
                    sys_scores.append(sys_score)

        return sys_scores

def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

def evaluate(args, model, data):
    entok = MosesTokenizer(lang='en')

    from argparse import Namespace

    new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                         params=args, model=model,
                         lower_case=model.args.lower_case,
                         tokenize=model.args.tokenize)
    s = FileSim()
    scores = s.score(new_args, batcher, data)

    #for i in range(len(scores)):
    #   print(data[i].strip() + "\t{0}".format(scores[i]))

    return scores

