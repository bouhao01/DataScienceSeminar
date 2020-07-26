from impl.datasets.dataset import Dataset
from gensim.models.doc2vec import TaggedDocument
import networkx as nx
from tqdm import tqdm
import impl.utils.config as config
import random


class CORA(Dataset):

    def __init__(self, content_filename, cites_filename):
        self.content_file = content_filename
        self.cites_file = cites_filename
        self.prepare_idx()
        self.prepare_tagged_docs()
        self.network = nx.read_edgelist(self.cites_file)

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
        return self.network


    def remove_random_edges(self, edges_percentage=5):
        number_of_edges_to_be_removed = int(edges_percentage*len(self.network.edges())/100)
        edges_to_be_removed = random.sample(self.network.edges(), number_of_edges_to_be_removed)

        self.network.remove_edges_from(edges_to_be_removed)
        print("Removed edges = {}".format(len(edges_to_be_removed)))

    def remove_iportant_edges_cc(self, edges_percentage=5):
        # Important edges with in relation with connected component only
        self.remove_iportant_edges(edges_percentage=edges_percentage, alpha=0.0, beta=1.0)

    def remove_iportant_edges_nd(self, edges_percentage=5):
        # Important edges with in relation with node degree only
        self.remove_iportant_edges(edges_percentage=edges_percentage, alpha=1.0, beta=0.0)

    def remove_iportant_edges(self, edges_percentage, alpha=0.4, beta=0.6):
        # Important edges are defined to be connected with nodes which have the less neighbors

        number_of_edges_to_be_removed = int(edges_percentage*len(self.network.edges())/100)

        edges_score = []

        initial_network_connected_components = nx.number_connected_components(self.network)

        for e in self.network.edges():
            edge_score_from_nodes = abs(self.network.degree(e[0]) - self.network.degree(e[1]))

            self.network.remove_edge(*e)
            edge_score_from_connected_components = nx.number_connected_components(self.network) - initial_network_connected_components
            self.network.add_edge(*e)

            edges_score.append((e, (alpha*edge_score_from_nodes + beta*edge_score_from_connected_components)))

        sorted(edges_score,  key=lambda x: x[1], reverse=True)
        edges_to_be_removed = [edge for edge, edges_score in edges_score[:number_of_edges_to_be_removed]]
        self.network.remove_edges_from(edges_to_be_removed)
        print("Removed edges = {}".format(len(edges_to_be_removed)))


    def remove_all_edges_of_random_nodes(self, edges_percentage=5):
        number_of_edges_to_be_removed = int(edges_percentage*len(self.network.edges())/100)

        nodes_to_apply_edges_delete = random.sample(self.network.nodes(), len(self.network.nodes()))

        # Delete edges
        removed_edges = 0
        for node in nodes_to_apply_edges_delete:
            node_edges = list(self.network.edges(node))
            self.network.remove_edges_from(node_edges)
            removed_edges += len(node_edges)
            if removed_edges >= number_of_edges_to_be_removed:
                break
        print("Removed edges = {}".format(removed_edges))

    def remove_all_edges_of_important_nodes(self, edges_percentage=5):
        # Important nodes are defined to have many neighbors
        number_of_edges_to_be_removed = int(edges_percentage*len(self.network.edges())/100)
        nodes_degree = [(node,self.network.degree(node)) for node in self.network.nodes()]
        # nodes_centrality = [(node,v) for node,v in nx.degree_centrality(network_citeseer).items()]

        nodes_degree = sorted(nodes_degree,  key=lambda x: x[1], reverse=True)
        # nodes_centrality = sorted(nodes_centrality,  key=lambda x: x[1], reverse=True)

        # Delete edges
        removed_edges = 0
        for node, degree in nodes_degree:
            node_edges = list(self.network.edges(node))
            self.network.remove_edges_from(node_edges)
            removed_edges += len(node_edges)
            if removed_edges >= number_of_edges_to_be_removed:
                break

        print("Removed edges = {}".format(removed_edges))