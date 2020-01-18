import random
from collections import defaultdict
import numpy as np
import numpy.random as rand
from tqdm import tqdm
import os
import multiprocessing as mp
import impl.utils.config as config
from impl.pairs.pair import Pair


class DWPairs(Pair):
    def __init__(self, dataset, walk_length=10, num_walks=80, power=0.75, window_size=5, neg_samples=5, sample=1e-3, min_count=5):
        self.G = dataset.get_network()
        self.window_size = window_size
        self.neg_samples = neg_samples
        self.vocab = None
        self.index2source = dataset.index2paper
        self.source2index = dataset.paper2index
        self.index2context = dataset.index2paper
        self.context2index = dataset.paper2index
        self.remove_unknown_nodes()
        self.generate_walks(num_walks, walk_length)
        self.build_vocab(min_count)
        self.ctx_weights = self.make_freq_table(power)
        self.neg_sample_buffer = []
        # downsample frequent words
        self.downsample_probs = np.zeros_like(self.ctx_weights)
        if sample > 0:
            self.downsample_probs = 1 - np.sqrt(sample/self.ctx_weights).clip(0, 1)
    
    def remove_unknown_nodes(self):
        filtered_nodes = [node for node in self.G.nodes if node in self.source2index]
        print(f"Removed {len(self.G.nodes) - len(filtered_nodes)} unknown nodes")
        self.G = self.G.subgraph(filtered_nodes)

    def generate_walks(self, num_walks, walk_length):
        self.sentences = []
        config.debug("DeepWalk: Generate walks")
        for cnt in tqdm(range(num_walks), desc="DeepWalk: Generate walks", disable=(not config.progress)):
            for node in self.G.nodes:
                self.sentences.append(random_walk(self.G, walk_length, node))

    def build_vocab(self, min_count):
        # word freqency
        raw_vocab = defaultdict(int)
        for sent in self.sentences:
            for word in sent:
                raw_vocab[word] += 1

        # only keep words that occur at least min_count times
        self.vocab = {k: v for k, v in raw_vocab.items() if v >= min_count}
        del raw_vocab

    def make_freq_table(self, power):
        # the unigram distribution raised to 0.75 empirically performed best as negative sampling distribution
        pow_frequency = np.array([self.vocab[k] if k in self.vocab else 1 for k in self.source2index.keys()])**power
        return pow_frequency / pow_frequency.sum()

    def sliding_window(self, words):
        for pos, word in enumerate(words):
            # sliding window (randomly reduced to give more weight to closeby words)
            reduction = np.random.randint(self.window_size)
            start = max(0, pos - self.window_size + reduction)
            for pos2, word2 in enumerate(words[start:(pos + self.window_size + 1 - reduction)], start):
                if pos2 != pos:
                    yield self.source2index[word], self.source2index[word2]

    def make_pairs(self, alg_index=None):
        self.alg_index = alg_index
        self.pairs = []

        config.debug("DeepWalk make pairs")
        workers = os.cpu_count()
        chunks = n_chunks(self.sentences, int(len(self.sentences) / workers))

        pool = mp.Pool(workers)
        results = pool.map(self.make_pairs_chunk, tqdm(chunks, desc="DeepWalk make pairs", disable=(not config.progress)))
        pool.close()
        pool.join()

        for res in results:
            self.pairs += res

    def make_pairs_chunk(self, chunk):
        pairs = []
        for sent in chunk:
            # randomly reject frequent words according to downsampling probability
            words = [w for w in sent if w in self.vocab and self.downsample_probs[self.source2index[w]] < random.random()]
            for source, context in self.sliding_window(words):
                pair = (source, context, self.negative_sampling(context))
                if self.alg_index is not None:
                    pair += (self.alg_index,)
                pairs.append(pair)
        return pairs


def random_walk(G, walk_length, start=None):
    path = [start]

    while len(path) < walk_length:
        cur = path[-1]
        neighbors = list(G.neighbors(cur))
        if len(neighbors) > 0:
            path.append(rand.choice(neighbors))
        else:
            break
    return [str(node) for node in path]


def n_chunks(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]
