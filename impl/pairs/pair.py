import numpy as np


class Pair():

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def negative_sampling(self, context):
        neg = []
        while len(neg) < self.neg_samples:
            try:
                negative = self.neg_sample_buffer.pop()
                # avoid that context word occurs in negative samples
                if negative != context:
                    neg.append(negative)
            except IndexError:
                self.neg_sample_buffer = list(np.random.choice(list(self.index2context.keys()), size=1000, replace=True, p=self.ctx_weights))
        return np.array(neg)
