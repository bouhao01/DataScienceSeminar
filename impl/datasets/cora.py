from impl.datasets.dataset import Dataset
from gensim.models.doc2vec import TaggedDocument
import networkx as nx
from tqdm import tqdm
import impl.utils.config as config


class CORA(Dataset):

    def __init__(self, content_filename, cites_filename):
        self.content_file = content_filename
        self.cites_file = cites_filename
        self.prepare_idx()
        self.prepare_tagged_docs()

    def prepare_idx(self):
        for paper_id, words_one_hot, class_label in self.get_content():
            if paper_id not in self.paper2index:
                self.paper2index[str(paper_id)] = len(self.paper2index)
            if class_label not in self.class2index:
                self.class2index[str(class_label)] = len(self.class2index)
            if self.word2index == {}:
                for w in range(len(words_one_hot)):
                    self.word2index[str(w)] = w

        self.index2paper = dict(zip(self.paper2index.values(), self.paper2index.keys()))
        self.index2class = dict(zip(self.class2index.values(), self.class2index.keys()))
        self.index2word = dict(zip(self.word2index.values(), self.word2index.keys()))

    def prepare_tagged_docs(self):
        self.tagged_docs = []
        config.debug("Get tagged docs")
        for paper_id, words_one_hot, class_label in tqdm(self.get_content(), desc="Get tagged docs", disable=(not config.progress)):
            words = [str(i) for i, x in enumerate(words_one_hot) if x == 1]
            self.tagged_docs.append(TaggedDocument(words, [paper_id]))

    def get_tagged_docs(self):
        return self.tagged_docs

    def get_content(self):
        num_words = None
        with open(self.content_file, 'r') as c:
            for line in c:
                content = line.strip().split('\t')
                paper_id = content[0]
                words_one_hot = list(map(int, content[1:-1]))
                class_label = content[-1]

                if num_words is None:
                    num_words = len(words_one_hot)
                if num_words != len(words_one_hot):
                    raise Exception("Error, different one-hot word lengths: ", len(words_one_hot), '!=', num_words)

                yield paper_id, words_one_hot, class_label

    def get_network(self):
        return nx.read_edgelist(self.cites_file)
