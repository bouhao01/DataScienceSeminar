from impl.pairs.pair import Pair
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import impl.utils.config as config


class D2VPairs(Pair):
    def __init__(self, dataset, power=0.75, neg_samples=5, sample=1e-3, min_count=5):
        self.dataset = dataset
        self.neg_samples = neg_samples
        self.vocab = None
        self.word_vocab = None
        self.index2source = dataset.index2paper
        self.source2index = dataset.paper2index
        self.index2context = dataset.index2word
        self.context2index = dataset.word2index
        self.build_vocab(min_count)
        self.ctx_weights = self.make_freq_table(power)
        self.neg_sample_buffer = []

    def build_vocab(self, min_count):
        self.vocab = defaultdict(int)
        # word freqency
        raw_vocab = defaultdict(int)
        for d in self.dataset.get_tagged_docs():
            for tag in d.tags:
                self.vocab[tag] += 1
            for word in d.words:
                raw_vocab[word] += 1
        # only keep words that occur at least min_count times
        self.word_vocab = {k: v for k, v in raw_vocab.items() if v >= min_count}
        del raw_vocab

    def make_freq_table(self, power):
        # the unigram distribution raised to 0.75 empirically performed best as negative sampling distribution
        pow_frequency = np.array([self.word_vocab[k] if k in self.word_vocab else 1 for k in self.context2index.keys()])**power
        return pow_frequency / pow_frequency.sum()

    def make_pairs(self, alg_index=None):
        self.pairs = []
        config.debug("Doc2Vec make pairs")
        for d in tqdm(self.dataset.get_tagged_docs(), desc="Doc2Vec make pairs", disable=(not config.progress)):
            words = [self.context2index[w] for w in d.words if w in self.word_vocab]

            for tag in d.tags:
                for w in words:
                    pair = (self.source2index[tag], w, self.negative_sampling(w))
                    if alg_index is not None:
                        pair += (alg_index,)
                    self.pairs.append(pair)

            del words
