## AutoML Experiments for Tabular Data

### Abstract
We aim to find a good way to automatically find architectures of neural networks that represent underlying connections in the Universe of the data. 
In Biology, for example, these are usually termed "Biologically-Informed neural networks" or sometimes "Visual Neural Networks". They are built to contain Gene-Pathway or Protein-Interaction information.

More generally, we consider discovery any hierarchical structure for which observed features (genes) are the leaf nodes.
This is different from _typical_ graph neural network problems, because the branch nodes (the pathways and processes) are not directly observed, but form hidden nodes or layers (including with skip connections) in the neural network.

We had a Poster for this purpose on the JGU Workshop for AI for Scientific Discovery: 
10.13140/RG.2.2.31376.42244
The code here was used for the results shown on this poster. 

### Workpackages
- [x] WP 1A: Build minimal running example of cancer-net (pytorch implementation of P-NET)
- [x] WP 1B: Get P-NET Dataset
- [ ] WP 2: Search for AutoML options, that provide single edge optimization

Wp3 experiments with different architectures:
- [x] PNET, but graph is randomized
- [x] Fully connected net
- [x] both with and without PNets 
- [ ] introduce spareseness by regularization (done but not finished)
- [ ] Introduce spareseness by mask: [ ] random [ ] with hierarcical network.
