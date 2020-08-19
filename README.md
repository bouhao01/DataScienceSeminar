## Data science seminar: Incomplete network embedding II
This project evaluates the effectiveness of three network embedding algorithms SINE, DeepWalk and Doc2Vec when the network presents different type of missing information namely:
- removing random edges
- removing important edges
- removing all edges of random nodes
- removing all edges of important nodes

where importance is defined by centrality measures.
Those algorithms will be used with three real-word data-sets: cora, citysee and dblp.
The evaluation of low-dimensional representations will be done using simple node classification where we train a classifier on a part of the embedded representation and test its performance on the rest of the data.

The implementation is devided into two main parts:
  1. Embed the dataset using SINE, DeepWalk and Doc2Vec after removing a percentage of edges.
  2. Evalute each embedded representation using a multi class logistic regression model.

To run the embedding:

> python generate_embeddings.py cora 5 50

The first argument represent the dataset to perform the embedding on (cora, citeseer or dblp).
After the name of the dataset, it comes a list of percentages of edges to be removed from the network.
After the program ends, it generates the following output in the cache dir of cora (the cache dir of the dataset):

```
2020-07-12-09-03/
    	├── 5
    	│   ├── remove_all_edges_of_important_nodes
    	│   │   ├── d2v
    	│   │   ├── dw
    	│   │   ├── sine
    	│   │   └── missing_edges_dataset.p
    	│   ├── remove_all_edges_of_random_nodes
    	│   │   ├── d2v
    	│   │   ├── dw
    	│   │   ├── sine
    	│   │   └── missing_edges_dataset.p
    	│   ├── remove_iportant_edges_cc
    	│   │   ├── d2v
    	│   │   ├── dw
    	│   │   ├── sine
    	│   │   └── missing_edges_dataset.p
    	│   ├── remove_iportant_edges_nd
    	│   │   ├── d2v
    	│   │   ├── dw
    	│   │   ├── sine
    	│   │   └── missing_edges_dataset.p
    	│   └── remove_random_edges
    	│       ├── d2v
    	│       ├── dw
    	│       ├── sine
    	│       └── missing_edges_dataset.p
    	└── 50
    	    ├── remove_all_edges_of_important_nodes
    	    │   ├── d2v
    	    │   ├── dw
    	    │   ├── sine
    	    │   └── missing_edges_dataset.p
    	    ├── remove_all_edges_of_random_nodes
    	    │   ├── d2v
    	    │   ├── dw
    	    │   ├── sine
    	    │   └── missing_edges_dataset.p
    	    ├── remove_iportant_edges_cc
    	    │   ├── d2v
    	    │   ├── dw
    	    │   ├── sine
    	    │   └── missing_edges_dataset.p
    	    ├── remove_iportant_edges_nd
    	    │   ├── d2v
    	    │   ├── dw
    	    │   ├── sine
    	    │   └── missing_edges_dataset.p
    	    └── remove_random_edges
    	        ├── d2v
    	        ├── dw
    	        ├── sine
    	        └── missing_edges_dataset.p
```
To run the evaluation:
> python evaluate_embeddings "/PATH_TO/2020-07-12-09-03/"

It takes one argument which is the path to the base dir of embeddings.
Example of output display:
```


		###################################################################################
		#  remove_all_edges_of_important_nodes with drop of 5
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6958  |  0.0141  |  0.7158  |  0.0138   |  0.7256  |  0.0116  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7864  |  0.0092  |  0.7937  |  0.0090   |  0.7951  |  0.0100  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8092  |  0.0087  |  0.8172  |  0.0085   |  0.8219  |  0.0082  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_all_edges_of_random_nodes with drop of 5
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6880  |  0.0085  |  0.7116  |  0.0088   |  0.7167  |  0.0083  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7856  |  0.0092  |  0.7954  |  0.0087   |  0.7942  |  0.0098  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8123  |  0.0096  |  0.8180  |  0.0102   |  0.8247  |  0.0100  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_iportant_edges_cc with drop of 5
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6932  |  0.0136  |  0.7144  |  0.0126   |  0.7227  |  0.0101  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7936  |  0.0149  |  0.8036  |  0.0136   |  0.8027  |  0.0149  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8172  |  0.0140  |  0.8218  |  0.0146   |  0.8252  |  0.0128  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_iportant_edges_nd with drop of 5
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6991  |  0.0140  |  0.7171  |  0.0145   |  0.7271  |  0.0119  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.8024  |  0.0134  |  0.8092  |  0.0141   |  0.8133  |  0.0135  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8053  |  0.0121  |  0.8095  |  0.0116   |  0.8195  |  0.0119  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_random_edges with drop of 5
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6922  |  0.0108  |  0.7152  |  0.0113   |  0.7218  |  0.0093  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7766  |  0.0097  |  0.7812  |  0.0106   |  0.7886  |  0.0086  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8097  |  0.0114  |  0.8152  |  0.0131   |  0.8202  |  0.0102  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_all_edges_of_important_nodes with drop of 50
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6872  |  0.0100  |  0.7104  |  0.0129   |  0.7149  |  0.0092  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.5760  |  0.0130  |  0.6338  |  0.0163   |  0.5929  |  0.0118  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.7194  |  0.0119  |  0.7349  |  0.0145   |  0.7428  |  0.0098  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_all_edges_of_random_nodes with drop of 50
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6949  |  0.0076  |  0.7170  |  0.0084   |  0.7257  |  0.0072  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.5957  |  0.0074  |  0.7144  |  0.0094   |  0.6001  |  0.0075  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.7615  |  0.0052  |  0.7704  |  0.0070   |  0.7776  |  0.0049  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_iportant_edges_cc with drop of 50
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6886  |  0.0067  |  0.7099  |  0.0064   |  0.7171  |  0.0075  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7554  |  0.0082  |  0.7729  |  0.0090   |  0.7745  |  0.0068  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.7991  |  0.0062  |  0.8083  |  0.0068   |  0.8134  |  0.0070  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_iportant_edges_nd with drop of 50
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6950  |  0.0093  |  0.7176  |  0.0067   |  0.7225  |  0.0088  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.7472  |  0.0101  |  0.7665  |  0.0095   |  0.7680  |  0.0094  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.8000  |  0.0084  |  0.8076  |  0.0099   |  0.8120  |  0.0075  |
		-----------------------------------------------------------------------------------



		###################################################################################
		#  remove_random_edges with drop of 50
		###################################################################################

		-----------------------------------------------------------------------------------
		|             |    macro_f1_score   |    precision_score   |      accuracy       |
		-----------------------------------------------------------------------------------
		|             |   AVG    |   STD    |   AVG    |   STD     |   AVG    |   STD    |
		-----------------------------------------------------------------------------------
		|  Doc2Vec    |  0.6906  |  0.0128  |  0.7101  |  0.0124   |  0.7177  |  0.0101  |
		-----------------------------------------------------------------------------------
		|  DeepWalk   |  0.6357  |  0.0107  |  0.6863  |  0.0072   |  0.6476  |  0.0066  |
		-----------------------------------------------------------------------------------
		|  SINE       |  0.7514  |  0.0109  |  0.7581  |  0.0108   |  0.7712  |  0.0104  |
		-----------------------------------------------------------------------------------
```
