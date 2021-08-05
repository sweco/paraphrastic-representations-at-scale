from scipy.stats import spearmanr
import os
import models
from argparse import Namespace
from sentence_transformers.readers import STSBenchmarkDataReader, STSDataReader
import csv
from score_sentence_pairs import evaluate


def slido_dataset(dataset='valid'):
    path = os.environ.get('SM_CHANNEL_SLIDO', 'data/slido-data')
    sts_reader = STSDataReader(path, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, min_score=0,
                               max_score=2, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    examples = sts_reader.get_examples(f'{dataset}.csv')

    pair_of_sentences = [ex.texts[0] + '\t' + ex.texts[1] for ex in examples]
    labels = [ex.label for ex in examples]

    return pair_of_sentences, labels


def sts_dataset(dataset='dev'):
    path = os.environ.get('SM_CHANNEL_STS', 'data/stsbenchmark')
    sts_reader = STSBenchmarkDataReader(path)
    examples = sts_reader.get_examples(f'sts-{dataset}.csv')

    pair_of_sentences = [ex.texts[0] + '\t' + ex.texts[1] for ex in examples]
    labels = [ex.label for ex in examples]

    return pair_of_sentences, labels


def eval(model, dataset, dev='dev'):
    sentences, labels = dataset(dev)
    preds = evaluate(model.args, model, sentences)

    res = spearmanr(labels, preds).correlation

    return res


if __name__ == '__main__':

    args = Namespace(gpu=0,
                     load_file='data/models/pt-model/model.para.lc.100.pt',
                     sp_model='data/models/sentencepiece/paranmt.model')

    model, _ = models.load_model(None, args, False)
    model.eval()

    print("STS", "Spearman rank cosine similarity:",
          eval(model, sts_dataset))

    print("Slido dev", "Spearman rank cosine similarity:",
          eval(model, slido_dataset, dev='dev-annot-backtr-balanced'))

    print("Slido valid", "Spearman rank cosine similarity:",
          eval(model, slido_dataset, dev='valid'))

    print("Slido test", "Spearman rank cosine similarity:",
          eval(model, slido_dataset, dev='test'))









