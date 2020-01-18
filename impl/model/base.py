import torch.nn as nn
import os
import torch.optim as optim
import torch


class ModelBase(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def to_cuda(self):
        self.u_embeddings = self.u_embeddings.cuda()
        self.v_embeddings = self.v_embeddings.cuda()


class NetworkBase():

    def __init__(self, cache=False, cache_file="", dim=100, alpha=0.025, dataset=None, iterations=5, workers=os.cpu_count()):
        self.cache_file = cache_file
        self.cache = cache
        self.dim = dim
        self.alpha = alpha
        self.dataset = dataset
        self.iterations = iterations
        self.workers = workers
        self.use_cuda = torch.cuda.is_available()
        self.embeddings = dict()
        self.losses = dict()

    def set_model(self, model):
        self.model = model
        if self.use_cuda:
            print("Using GPU")
            self.model.cuda()
            self.model.to_cuda()
        else:
            print("Using CPU")
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, weight_decay=1e-6)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_steps, gamma=self.lr_gamma)

        self.set_optimizer()
        if not self.restore_cache():
            self.train()
            self.save_cache()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda())

    def lr_lambda(self):
        return lambda epoch: 1 - epoch/self.iterations

    def restore_cache(self):
        if not self.cache or not os.path.exists(self.cache_file):
            return False
        print("Loading cached embeddings...")
        self.embeddings, self.losses = torch.load(self.cache_file)
        return True

    def save_cache(self):
        if not self.cache:
            return
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Saving embeddings to cache...")
        torch.save((self.embeddings, self.losses), self.cache_file)

    def report_values(self, epoch, avg_loss):
        if self.use_cuda:
            avg_loss = avg_loss.cpu()
        self.embeddings[str(epoch + 1)] = self.get_embedding()
        self.losses[str(epoch + 1)] = avg_loss.item()

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return float(param_group['lr'])

    def train(self):
        pass

    def get_embedding(self):
        pass

    def most_similar(self, to_test, top_n=10):
        pass
