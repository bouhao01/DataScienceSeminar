from impl.datasets.cora import CORA
from impl.datasets.citeseer import Citeseer
from impl.datasets.dblp import DBLP

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import impl.utils.config as config
from matplotlib.backends.backend_pdf import PdfPages

import os
import sys
import pickle
import torch





def eval_model(model, dataset, folds=10, test_size=0.5):
    X = []
    y = []
    macro_f1_scores = []
    precision_scores = []
    accuracies = []

    for paper_id, words_one_hot, class_label in dataset.get_content():
        if paper_id in model:
            X.append(model[paper_id])
            y.append(class_label)
        else:
            print(f"node {paper_id} has been removed")

    clf = LogisticRegression(solver='liblinear', multi_class='ovr')

    for i in range(folds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        macro_f1_score = metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred), average='macro')
        precision_score = metrics.precision_score(y_test, y_pred, labels=np.unique(y_pred), average='macro')
        accuracy = metrics.accuracy_score(y_test, y_pred)

        macro_f1_scores.append(macro_f1_score)
        precision_scores.append(precision_score)
        accuracies.append(accuracy)

    return [[np.mean(macro_f1_scores), np.std(macro_f1_scores)],
            [np.mean(precision_scores), np.std(precision_scores)],
            [np.mean(accuracies), np.std(accuracies)]]

def main(args):
    if 1 != len(args):
        print("Missing the base dir's path of the embeddings")
        print("Ex: python evaluate_embeddings \"PATH\"")
        return

    ############################################################################################
    # Loading all tests
    ############################################################################################
    BASE_DIR = args[0]
    tests = []
    if os.path.isdir(BASE_DIR):
        for drop in os.listdir(BASE_DIR):
            if os.path.isdir(BASE_DIR + drop):
                for test in os.listdir(BASE_DIR + drop + "/"):
                    # Add test
                    test_item = dict()
                    test_item['drop'] = drop
                    test_item['test_name'] = test
                    test_item['test_result'] = BASE_DIR + drop + "/" + test + "/test_result.txt"
                    for item in os.listdir(BASE_DIR + drop + "/" + test + "/"):
                        if "dataset" in item:
                            test_item['dataset'] = BASE_DIR + drop + "/" + test + "/" + item
                        elif "d2v" in item:
                            test_item['doc2vec'] = BASE_DIR + drop + "/" + test + "/" + item
                        elif "dw" in item:
                            test_item['deepwalk'] = BASE_DIR + drop + "/" + test + "/" + item
                        elif "sine" in item:
                            test_item['sine'] = BASE_DIR + drop + "/" + test + "/" + item
                    tests.append(test_item)

        ############################################################################################
        # Run evaluation on each test and display results
        ############################################################################################
        global_report = ""
        for test in tests:
            with open(test['dataset'], 'rb') as f:
                dataset = pickle.load(f)

            d2v_embeddings, d2v_losses = torch.load(test['doc2vec'])
            d2v_last_embedding = d2v_embeddings['10']
            d2v_evaluation = eval_model(d2v_last_embedding, dataset)

            dw_embeddings, dw_losses = torch.load(test['deepwalk'])
            dw_last_embedding = dw_embeddings['10']
            dw_evaluation = eval_model(dw_last_embedding, dataset)

            sin_embeddings, sin_losses = torch.load(test['sine'])
            sin_last_embedding = sin_embeddings['10']
            sin_evaluation = eval_model(sin_last_embedding, dataset)

            res = "\n\n\n###################################################################################\n" \
            "#  {} with drop of {}".format(test['test_name'], test['drop']) + "\n" \
            "###################################################################################\n\n" \
            "-----------------------------------------------------------------------------------\n" \
            "|             |    macro_f1_score   |    precision_score   |      accuracy       |" + "\n" \
            "-----------------------------------------------------------------------------------\n" \
            "|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |" + "\n" \
            "-----------------------------------------------------------------------------------\n" \
            "|  Doc2Vec    |  {:.4f}  |  {:.4f}  |  {:.4f}  |  {:.4f}   |  {:.4f}  |  {:.4f}  |".format(*d2v_evaluation[0], *d2v_evaluation[1], *d2v_evaluation[2]) + "\n" \
            "-----------------------------------------------------------------------------------\n" \
            "|  DeepWalk   |  {:.4f}  |  {:.4f}  |  {:.4f}  |  {:.4f}   |  {:.4f}  |  {:.4f}  |".format(*dw_evaluation[0], *dw_evaluation[1], *dw_evaluation[2]) + "\n" \
            "-----------------------------------------------------------------------------------\n" \
            "|  SINE       |  {:.4f}  |  {:.4f}  |  {:.4f}  |  {:.4f}   |  {:.4f}  |  {:.4f}  |".format(*sin_evaluation[0], *sin_evaluation[1], *sin_evaluation[2]) + "\n" \
            "-----------------------------------------------------------------------------------\n" \

            test_result = open(test['test_result'], 'w')
            test_result.write(res)
            test_result.close()
            global_report += res
            print(res)

        with open(BASE_DIR + "global_report.txt", 'w') as f:
            f.write(global_report)

    else:
        print("there is no dir in {}".format(BASE_DIR))






if __name__ == "__main__":
    main(sys.argv[1:])
