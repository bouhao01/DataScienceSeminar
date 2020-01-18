import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from impl.model.base import NetworkBase, ModelBase
import impl.utils.config as config
from itertools import cycle, islice
from impl.utils.eval import eval_model
import torch.optim as optim


class JCEModel(ModelBase):

    def __init__(self, emb_size_u, emb_sizes_v, emb_dimension):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.u_emb_shares = [emb_dimension // len(emb_sizes_v) + (1 if x < emb_dimension % len(emb_sizes_v) else 0) for x in range(len(emb_sizes_v))]
        self.u_embeddings = nn.ModuleList([nn.Embedding(emb_size_u, size) for size in self.u_emb_shares])
        self.v_embeddings = nn.ModuleList([nn.Embedding(size, emb_dimension) for size in emb_sizes_v])
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        # center words weight randomly initialized in range -0.5/emb_dim, 0.5/emb_dim
        for u_embedding in self.u_embeddings:
            u_embedding.weight.data.uniform_(-initrange, initrange)
        # context words initialized with zeroes
        for v_embedding in self.v_embeddings:
            v_embedding.weight.data.uniform_(-0, 0)

    def get_embedding(self):
        return torch.cat([u_emb.weight for u_emb in self.u_embeddings], dim=1)

    def get_u_embedding(self, pos_u):
        return torch.cat([u_embedding(pos_u) for u_embedding in self.u_embeddings], dim=1)

    # pos_u: [batch_size]
    # pos_v: [batch_size]
    # neg_v: [batch_size, neg_sampling_count]
    def forward(self, pos_u, pos_v, neg_v, alg_index):
        # emb_u,v: [batch_size, emb_dim]
        # emb_neg_v: [batch_size, neg_size, emb_dim]
        emb_u = self.get_u_embedding(pos_u)
        emb_v = self.v_embeddings[alg_index](pos_v)
        emb_neg_v = self.v_embeddings[alg_index](neg_v)

        pos_score = torch.sum(torch.mul(emb_u, emb_v).squeeze(), dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))


class JCE(NetworkBase):
    def __init__(self, data, batch_sizes=[50, 500], disable_grad=True, **kwargs):
        super().__init__(**kwargs)

        it = iter(data)
        the_len = len(next(it).index2source)
        if not all(len(l.index2source) == the_len for l in it):
            raise ValueError('not all pairs have same length!')

        v_lengths = [len(alg.index2context) for alg in data]

        self.data = data
        self.batch_sizes = batch_sizes
        self.disable_grad = disable_grad
        self.set_model(JCEModel(len(self.data[0].index2source), v_lengths, self.dim))

    def set_optimizer(self):
        self.optimizers = [optim.Adam(self.model.parameters(), lr=self.alpha) for d in self.data]
        self.schedulers = [optim.lr_scheduler.LambdaLR(o, lr_lambda=self.lr_lambda()) for o in self.optimizers]
        self.optimizer = self.optimizers[0]  # for debugging purposes

    def train(self):
        for epoch in range(self.iterations):
            for i, data in enumerate(self.data):
                data.make_pairs(alg_index=i)
            loaders = [DataLoader(pairs, batch_size=self.batch_sizes[i], shuffle=True,
                                  num_workers=self.workers, pin_memory=True) for i, pairs in enumerate(self.data)]
            batches = round_robin([iter(loader) for loader in loaders])

            total_batches = np.sum([len(loader) for loader in loaders])
            tenth = int(total_batches/10)
            epoch_loss = 0
            epoch_batches = 0
            avg_loss = 0

            loop = tqdm(enumerate(batches), total=total_batches, disable=(not config.progress))
            for i, (pos_u, pos_v, neg_v, alg_index) in loop:
                alg_index = alg_index.data[0]  # Reduce [batch_size] to int
                epoch_batches += 1

                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()
                    alg_index = alg_index.cuda()

                # Disable backward autograd for all embeddings expect the one of the current algorithm
                if self.disable_grad:
                    optimizer = self.optimizers[alg_index]
                    disabled_embeddings = []
                    for name, param in self.model.named_parameters():
                        if name.startswith('u_embeddings') and not name.startswith(f'u_embeddings.{alg_index}'):
                            param.requires_grad = False
                            disabled_embeddings.append(param)
                else:
                    # for SINE (not disabling grad) we select the same optimizer for everything
                    optimizer = self.optimizers[0]

                # Zero gradients
                optimizer.zero_grad()
                # Execute forward pass and get loss
                loss = self.model.forward(pos_u, pos_v, neg_v, alg_index)
                epoch_loss += loss
                # Execute backward pass
                loss.backward()
                optimizer.step()

                # Re-enable all disabled embeddings for next batch
                if self.disable_grad:
                    for disabled_emb in disabled_embeddings:
                        disabled_emb.requires_grad = True

                if i % tenth == 0:
                    avg_loss = epoch_loss/epoch_batches
                    config.debug(f'Epoch {epoch+1}/{self.iterations} - {int(i/total_batches * 100)}%')

                if config.progress:
                    loop.set_description(f'Epoch {epoch+1}/{self.iterations}, Total Loss {epoch_loss.round()}, Avg. Loss {avg_loss.round()}' +
                                         f', LR {self.get_current_lr()}')
            config.debug(f'Epoch {epoch+1}/{self.iterations} - 100%')
            loop.close()
            del loop, batches, loaders
            for scheduler in self.schedulers:
                scheduler.step()
            self.report_values(epoch, epoch_loss/epoch_batches)
            if self.dataset:
                print("Evaluation: ", eval_model(self.get_embedding(), self.dataset, folds=10))

    def get_embedding(self):
        # Get weights of center words => Actual embeddings
        e = dict()
        embedding = np.array([weight.cpu().data.numpy() for weight in self.model.get_embedding()])
        for i in range(len(self.data[0].index2source)):
            e[self.data[0].index2source[i]] = embedding[i]
        return e

    def most_similar(self, to_test, top_n=10):
        test_tensor = torch.LongTensor([self.data[0].source2index[to_test]])
        emb_weight = self.model.get_embedding().cpu()
        if self.use_cuda:
            test_tensor = test_tensor.cuda()
        emb_test = self.model.get_u_embedding(test_tensor).cpu()
        score = torch.mm(emb_weight.data, torch.t(emb_test))
        norms_emb = torch.norm(emb_weight, dim=1)
        normalization_factors = norms_emb * torch.norm(emb_test)
        scores = score.squeeze()/normalization_factors
        values, indices = scores.sort(descending=True)
        values = values.detach().numpy()
        indices = indices.detach().numpy()
        if top_n < 0 or top_n > len(self.data[0].index2source):
            top_n = len(self.data[0].index2source)
        for i in range(top_n):
            print(self.data[0].index2source[indices[i]], ':', values[i])


def round_robin(iterables):
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))
