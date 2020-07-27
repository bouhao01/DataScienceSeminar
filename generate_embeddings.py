from impl.datasets.cora import CORA
from impl.datasets.citeseer import Citeseer
from impl.datasets.dblp import DBLP
from impl.pairs.doc2vec import D2VPairs
from impl.pairs.deepwalk import DWPairs
from impl.model.skipgram import SGNS
from impl.model.jce import JCE
from impl.utils.eval import eval_model, plot_evals

from gensim.models import Word2Vec, doc2vec
import numpy as np
import psutil
import os
import datetime
import pickle
import sys
import json
import copy






params = None
complete_dataset = None
drop_percentages = []
test_cases = [    "remove_random_edges"
                , "remove_iportant_edges_cc"
                , "remove_iportant_edges_nd"
                , "remove_all_edges_of_random_nodes"
                , "remove_all_edges_of_important_nodes"]

starting_run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
BASE_CACHE_DIR = ""
TIMESTAMP_CACHE_DIR = ""



def init(args):
    global params, drop_percentages, BASE_CACHE_DIR, TIMESTAMP_CACHE_DIR, complete_dataset

    with open('params.json') as json_file:
        all_params = json.load(json_file)

    if 2 > len(args):
        print("Please enter the name of the dataset in addition to the list of edges removal percentages.")
        print("Ex: python generate_embeddings cora 5 10 50")
        return False

    if "cora" == args[0]:
        params = all_params["cora"]
        complete_dataset = CORA(content_filename='./datasets/cora/cora.content', cites_filename='./datasets/cora/cora.cites')
    elif "citeseer" == args[0]:
        params = all_params["citeseer"]
        complete_dataset = Citeseer(content_filename='./datasets/citeseer/citeseer.content', cites_filename='./datasets/citeseer/citeseer.cites')
    elif "dblp" == args[0]:
        params = all_params["dblp"]
        complete_dataset = DBLP(content_filename='./datasets/dblp/dblp.content', cites_filename='./datasets/dblp/dblp.cites')
    else:
        print("Dataset name should be cora, citeseer or dblp")
        print("Ex: python generate_embeddings cora 5 10 50")
        return False

    params["sine"]["batch_sizes"] = [params["d2v"]["model"]["batch_size"], params["dw"]["model"]["batch_size"]]

    for drop in args[1:]:
        drop_percentages.append(int(drop))

    BASE_CACHE_DIR = "./cache/{}".format(args[0])
    TIMESTAMP_CACHE_DIR = "{}/{}".format(BASE_CACHE_DIR, starting_run_date)

    os.system("mkdir ./cache 2> /dev/null")
    os.system("mkdir {} 2> /dev/null".format(BASE_CACHE_DIR))
    os.system("mkdir {} 2> /dev/null".format(TIMESTAMP_CACHE_DIR))

    print("Parameters are well initialized")
    return True

def main():

    for drop_percentage in drop_percentages:
        os.system("mkdir {}/{} 2> /dev/null".format(TIMESTAMP_CACHE_DIR, drop_percentage))
        if 0 == drop_percentage:
            # run on Full dataset
            test_to_be_run = ["full_dataset"]
        else :
            test_to_be_run = test_cases
        for test in test_to_be_run:
            print("Performing test {} with drop_percentage = {}".format(test, drop_percentage))
            dataset = copy.copy(complete_dataset)
            # do not perform any edge removal when running on full dataset
            if "full_dataset" != test:
                print("Removing process ...")
                remove_method = getattr(dataset, test)
                remove_method(drop_percentage)

            os.system("mkdir {}/{}/{} 2> /dev/null".format(TIMESTAMP_CACHE_DIR, drop_percentage, test))
            cache_dir = "{}/{}/{}".format(TIMESTAMP_CACHE_DIR, drop_percentage, test)

            params["d2v"]["model"]["cache_file"] = cache_dir + "/d2v"
            params["dw"]["model"]["cache_file"] = cache_dir + "/dw"
            params["sine"]["cache_file"] = cache_dir + "/sine"

            ################################################################
            # Saving dataset
            ################################################################
            print("saving missing dataset in {}/missing_edges_dataset.p".format(cache_dir))
            pickle.dump(dataset, open("{}/missing_edges_dataset.p".format(cache_dir), "wb"))

            ################################################################
            # Making pairs
            ################################################################
            d2vpairs = D2VPairs(dataset=dataset, **params["d2v"]["pairs"])
            dwpairs = DWPairs(dataset=dataset, **params["dw"]["pairs"])

            #############################################
            # Doc2Vec
            #############################################
            print("Running Doc2Vec ...")
            d2vmodel = SGNS(d2vpairs, **params["d2v"]["model"])
            d2vEmb = d2vmodel.embeddings[str(params["d2v"]["model"]["iterations"])]
            print("Eval D2V: ", eval_model(d2vEmb, dataset=dataset))

            #############################################
            # DeepWalk
            #############################################
            print("Running DeepWalk ...")
            dwmodel = SGNS(dwpairs, **params["dw"]["model"])
            dwEmb = dwmodel.embeddings[str(params["dw"]["model"]["iterations"])]
            print("Eval DW: ", eval_model(dwEmb, dataset=dataset))

            #############################################
            # SINE
            #############################################
            print("Running SINE ...")
            sinemodel = JCE(data=[d2vpairs, dwpairs], disable_grad=False, **params["sine"])
            sineEmb = sinemodel.embeddings[str(params["sine"]["iterations"])]
            print("Eval JCE (SINE): ", eval_model(sineEmb, dataset=dataset))

if __name__ == '__main__':
    if init(sys.argv[1:]):
        main()
