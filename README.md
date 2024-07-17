## AutoML Experiments for Tabular Data

### Abstract
We aim to find a good way to automatically find architectures of neural networks that represent underlying connections in the Universe of the data. 
In Biology, for example, these are usually termed "Biologically-Informed neural networks" or sometimes "Visual Neural Networks". They are built to contain Gene-Pathway or Protein-Interaction information.

More generally, we consider discovery any hierarchical structure for which observed features (genes) are the leaf nodes.
This is different from _typical_ graph neural network problems, because the branch nodes (the pathways and processes) are not directly observed, but form hidden nodes or layers (including with skip connections) in the neural network.

### Workpackages
- WP 1A: Build minimal running example of cancer-net (pytorch implementation of P-NET)
- WP 1B: Get P-NET Dataset
- WP 2: Search for AutoML options, that provide single edge optimization