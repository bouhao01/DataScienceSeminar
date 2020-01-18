from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import impl.utils.config as config
from matplotlib.backends.backend_pdf import PdfPages


def eval_model(model, dataset, folds=20, test_size=0.5):
    X = []
    y = []
    for paper_id, words_one_hot, class_label in dataset.get_content():
        X.append(model[paper_id])
        y.append(class_label)
    clf = LogisticRegression(solver='liblinear', multi_class='ovr')
    scores = []
    accuracies = []
    for i in range(folds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred), average='macro')
        accuracy = metrics.accuracy_score(y_test, y_pred)
        scores.append(score)
        accuracies.append(accuracy)
    return [(np.mean(scores), np.std(scores)), (np.mean(accuracies), np.std(accuracies))]


def plot_evals(model, dataset, title, folds=20, test_size=0.5):
    pp = PdfPages(f'./out/figures/{title}.pdf')
    x = [int(val) for val in model.embeddings.keys()]
    y_loss = []
    y_acc = []
    y_macro = []
    config.debug("Evaluating model")
    for epoch in tqdm(model.embeddings.keys(), desc="Evaluating model", disable=(not config.progress)):
        macro, acc = eval_model(model.embeddings[epoch], folds=folds, test_size=test_size, dataset=dataset)
        y_loss.append(model.losses[epoch])
        y_acc.append(acc[0])
        y_macro.append(macro[0])

    # => Accuracy/Loss
    fig1, ax1 = plt.subplots()
    ax1.set_title(title + " (Loss/Accuracy)")
    color = 'tab:blue'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('average loss (training)', color=color)
    ax1.plot(x, y_loss, color=color, marker='s')
    ax1.tick_params(axis='y', labelcolor=color)

    ax1_2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax1_2.set_ylabel('mean accuracy (classification)', color=color)  # we already handled the x-label with ax1
    ax1_2.plot(x, y_acc, color=color, marker='o')
    ax1_2.tick_params(axis='y', labelcolor=color)

    fig1.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    fig1.savefig(pp, format='pdf')

    # => MacroF1
    fig2, ax2 = plt.subplots()
    ax2.set_title(title + " (Macro F1)")
    ax2.plot(x, y_macro, marker='^')
    plt.show()
    fig2.savefig(pp, format='pdf')

    pp.close()
