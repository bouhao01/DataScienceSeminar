from impl.datasets.cora import CORA


class DBLP(CORA):

    def get_content(self):
        with open(self.content_file, 'r') as c:
            for line in c:
                content = line.strip().split('\t')
                paper_id = content[0]
                words_one_hot = list(map(int, content[1:-2]))
                class_label = content[-2]
                # year = content[-1]

                yield paper_id, words_one_hot, class_label
