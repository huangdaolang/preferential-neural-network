# Preferential Bayesian Neural Network for Expert Knowledge Elicitation

This is PBNN source code, a novel preference learning architecture which is based on Siamese neural network. it not only learns the instance preference relationship, but is also capable of capturing the latent function shape. As working with humans implies a limited number of queries, we use active learning to sequentially ask the most informative queries from the expert.

The experiments are did on three regreesion datasets: *Boston housing*, *Machine CPU* and *Pyrimidine*. And can use three active learning strategies to query for the next data points: Random, Maximum entropy and BALD.

This repository only contains the expert knowledge elicitation but not Bayesian optimization. The full codes of our expert knowledge-augmented BO will release after the paper acceptance.