from impl.datasets.cora import CORA
from impl.datasets.citeseer import Citeseer
from impl.datasets.dblp import DBLP
from impl.pairs.deepwalk import DWPairs
from impl.pairs.doc2vec import D2VPairs
from impl.model.skipgram import SGNS
from impl.model.jce import JCE
from impl.utils.eval import eval_model
from ray import tune
import ray
import impl.utils.config as config

ray.init(num_gpus=2)


def dw(name, dataset, resume, cfg):

    def train_dw(config):
        adjust_config()
        data = dataset()
        dwpairs = DWPairs(dataset=data, **config["pairs"])
        dwemb = SGNS(dwpairs, dataset=data, **config["model"])
        last_iter = str(config["model"]["iterations"])
        macro, acc = eval_model(dwemb.embeddings[last_iter], dataset=data)
        tune.track.log(f1_macro=macro, accuracy=acc, avg_loss=dwemb.losses[last_iter])

    analysis = tune.run(
        train_dw,
        name=name,
        resume=resume,
        resources_per_trial={"gpu": 0.2},
        config=cfg
    )

    analysis.dataframe()[["date", "config/model", "config/pairs", "f1_macro", "accuracy", "avg_loss"]].sort_values(
        by="f1_macro", axis=0, ascending=False
    ).to_csv(f'/root/bachelor-thesis/code/gridsearch-{name}.csv')


def d2v(name, dataset, resume, cfg):

    def train_d2v(config):
        adjust_config()
        data = dataset()
        d2vpairs = D2VPairs(dataset=data, **config["pairs"])
        d2vemb = SGNS(d2vpairs, dataset=data, **config["model"])
        last_iter = str(config["model"]["iterations"])
        macro, acc = eval_model(d2vemb.embeddings[last_iter], dataset=data)
        tune.track.log(f1_macro=macro, accuracy=acc, avg_loss=d2vemb.losses[last_iter])

    analysis = tune.run(
        train_d2v,
        name=name,
        resume=resume,
        resources_per_trial={"gpu": 0.2},
        config=cfg
    )

    analysis.dataframe()[["date", "config/model", "config/pairs", "f1_macro", "accuracy", "avg_loss"]].sort_values(
        by="f1_macro", axis=0, ascending=False
    ).to_csv(f'/root/bachelor-thesis/code/gridsearch-{name}.csv')


def jce(name, dataset, resume, cfg):

    def train_jce(config):
        adjust_config()
        data = dataset()
        d2vpairs = D2VPairs(dataset=data, **config["pairs_d2v"])
        dwpairs = DWPairs(dataset=data, **config["pairs_dw"])
        jceemb = JCE(data=[d2vpairs, dwpairs], dataset=data, **config["model"])
        last_iter = str(config["model"]["iterations"])
        macro, acc = eval_model(jceemb.embeddings[last_iter], dataset=data)
        tune.track.log(f1_macro=macro, accuracy=acc, avg_loss=jceemb.losses[last_iter])

    analysis = tune.run(
        train_jce,
        name=name,
        resume=resume,
        resources_per_trial={"gpu": 0.20},
        config=cfg
    )

    analysis.dataframe()[["date", "config/model", "f1_macro", "accuracy", "avg_loss"]].sort_values(
        by="f1_macro", axis=0, ascending=False
    ).to_csv(f'/root/bachelor-thesis/code/gridsearch-{name}.csv')


def cora():
    return CORA(content_filename='/root/bachelor-thesis/code/datasets/cora/cora.content',
                cites_filename='/root/bachelor-thesis/code/datasets/cora/cora.cites')


def citeseer():
    return Citeseer(content_filename='/root/bachelor-thesis/code/datasets/citeseer/citeseer.content',
                    cites_filename='/root/bachelor-thesis/code/datasets/citeseer/citeseer.cites')


def dblp():
    return DBLP(content_filename='/root/bachelor-thesis/code/datasets/dblp/dblp.content',
                cites_filename='/root/bachelor-thesis/code/datasets/dblp/dblp.cites')


def adjust_config():
    config.messages = True
    config.progress = False


dw("dw-cora", cora, False, {
   "pairs": {
       "neg_samples": 10,
       "sample": 0.001,
       "min_count": 5,
       "walk_length": 40,
       "num_walks": 10,
       "window_size": 5
   },
   "model": {
       "dim": 100,
       "alpha": tune.grid_search([0.001, 0.025, 0.005]),
       "iterations": 10,
       "batch_size": 500,
       "cache_file": "./cache/cora/dw"
   }
})

# jce("jce-cora", cora, False, {
#     "pairs_d2v": {  # Fixed
#         "neg_samples": 10,
#         "sample": 0,
#         "min_count": 5
#     },
#     "pairs_dw": {  # Fixed
#         "neg_samples": 10,
#         "sample": 0.001,
#         "min_count": 5,
#         "walk_length": 40,
#         "num_walks": 10,
#         "window_size": 5
#     },
#     "model": {
#         "dim": 200,
#         "alpha": tune.grid_search([0.001, 0.0005, 0.025, 0.005]),
#         "lr_gamma": 1,
#         "lr_steps": 1,
#         "iterations": 10,
#         "batch_sizes": [4, 500]
#     }
# })


# dw("dw-citeseer", citeseer, False, {
#     "pairs": {
#         "neg_samples": 10,
#         "sample": 0.001,
#         "min_count": 5,
#         "walk_length": 40,
#         "num_walks": 10,
#         "window_size": 5
#     },
#     "model": {
#         "dim": 100,
#         "alpha": 0.001,
#         "lr_gamma": tune.grid_search([0.75, 0.5, 0.1]),
#         "lr_steps": tune.grid_search([1, 2, 3]),
#         "iterations": 10,
#         "batch_size": tune.grid_search([500, 2000])
#     }
# })
