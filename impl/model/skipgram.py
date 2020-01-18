import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from impl.model.base import NetworkBase, ModelBase
import impl.utils.config as config
from impl.utils.eval import eval_model


class SkipGramModel(ModelBase):

    def __init__(self, emb_size_u, emb_size_v, emb_dimension):
        super().__init__()
        # Embedding dimension
        self.emb_dimension = emb_dimension
        # Lookup tables for center (u) and context (v) words
        # center words => hidden layer
        self.u_embeddings = nn.Embedding(emb_size_u, emb_dimension)
        # context words => output layer
        self.v_embeddings = nn.Embedding(emb_size_v, emb_dimension)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        # center words weight randomly initialized in range -0.5/emb_dim, 0.5/emb_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        # context words initialized with zeroes
        self.v_embeddings.weight.data.uniform_(-0, 0)

    # pos_u: [batch_size]
    # pos_v: [batch_size]
    # neg_v: [batch_size, neg_sampling_count]
    def forward(self, pos_u, pos_v, neg_v):
        # emb_u,v: [batch_size, emb_dim]
        # emb_neg_v: [batch_size, neg_size, emb_dim]
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        pos_score = torch.sum(torch.mul(emb_u, emb_v).squeeze(), dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))


class SGNS(NetworkBase):
    def __init__(self, data, batch_size=500, **kwargs):
        super().__init__(**kwargs)

        self.data = data
        self.batch_size = batch_size
        self.set_model(SkipGramModel(len(self.data.index2source), len(self.data.index2context), self.dim))

    def train(self):
        for epoch in range(self.iterations):
            self.data.make_pairs()
            loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)

            total_batches = len(loader)
            tenth = int(total_batches/10)
            epoch_loss = 0
            epoch_batches = 0
            avg_loss = 0

            loop = tqdm(enumerate(loader), total=total_batches, disable=(not config.progress))
            for i, (pos_u, pos_v, neg_v) in loop:
                epoch_batches += 1

                if self.use_cuda:
                    pos_u = pos_u.cuda()
                    pos_v = pos_v.cuda()
                    neg_v = neg_v.cuda()

                # Zero gradients
                self.optimizer.zero_grad()
                # Execute forward pass and get loss
                loss = self.model.forward(pos_u, pos_v, neg_v)
                epoch_loss += loss
                # Execute backward pass
                loss.backward()
                self.optimizer.step()

                if i % tenth == 0:
                    avg_loss = epoch_loss/epoch_batches
                    config.debug(f'Epoch {epoch+1}/{self.iterations} - {int(i/total_batches * 100)}%')
       
                if config.progress:
                    loop.set_description(f'Epoch {epoch+1}/{self.iterations}, Total Loss {epoch_loss.round()}, Avg. Loss {avg_loss.round()}' +
                                         f', LR {self.get_current_lr()}')
            config.debug(f'Epoch {epoch+1}/{self.iterations} - 100%')
            loop.close()
            del loader, loop
            self.scheduler.step()
            self.report_values(epoch, epoch_loss/epoch_batches)
            if self.dataset:
                print("Evaluation: ", eval_model(self.get_embedding(), self.dataset, folds=10))

    def get_embedding(self):
        # Get weights of center words => Actual embeddings
        e = dict()
        embedding = self.model.u_embeddings.weight.cpu().data.numpy()
        for i in range(len(self.data.index2source)):
            e[self.data.index2source[i]] = embedding[i]
        return e

    def most_similar(self, to_test, top_n=10):
        test_tensor = torch.LongTensor([self.data.source2index[to_test]])
        emb_weight = self.model.u_embeddings.weight.cpu()
        if self.use_cuda:
            test_tensor = test_tensor.cuda()
        emb_test = self.model.u_embeddings(test_tensor).cpu()
        score = torch.mm(emb_weight.data, torch.t(emb_test))
        norms_emb = torch.norm(emb_weight, dim=1)
        normalization_factors = norms_emb * torch.norm(emb_test)
        scores = score.squeeze()/normalization_factors
        values, indices = scores.sort(descending=True)
        values = values.detach().numpy()
        indices = indices.detach().numpy()
        if top_n < 0 or top_n > len(self.data.index2source):
            top_n = len(self.data.index2source)
        for i in range(top_n):
            print(self.data.index2source[indices[i]], ':', values[i])
