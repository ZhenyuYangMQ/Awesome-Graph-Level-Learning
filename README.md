# Awesome Graph Level Learning
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) ![GitHub stars](https://img.shields.io/github/stars/ZhenyuYangMQ/Awesome-Graph-Level-Learning?color=yellow&label=Stars) ![GitHub forks](https://img.shields.io/github/forks/ZhenyuYangMQ/Awesome-Graph-Level-Learning?color=blue&label=Forks) 

A collection of papers, implementations, datasets, and tools for graph-level learning.


- [Awesome Graph-level Learning](#a-timeline-of-graph-level-learning)
  - [Surveys](#surveys)
  - [Traditional Graph-level Learning](#traditional-graph-level-learning)
    - [Graph Kernels](#graph-kernels)
      - [Message Passing Kernels](#message-passing-kernels)
      - [Shortest Path Kernels](#shortest-path-kernels)
      - [Random Walk Kernels](#random-walk-kernels)
      - [Optimal Assignment Kernels](#optimal-assignment-kernels)
      - [Subgraph Kernels](#subgraph-kernels)
    - [Subgraph Mining](#subgraph-mining)
      - [Frequent Subgraph Mining](#frequent-subgraph-mining)
      - [Discriminative Subgraph Mining](#discriminative-subgraph-mining)
    - [Graph Embedding](#graph-embedding)
      - [Deterministic Graph Embedding](#deterministic-graph-embedding)
      - [Learnable Graph Embedding](#learnable-graph-embedding)
  - [Graph-Level Deep Neural Networks (GL-DNNs)](#graph-Level-deep-neural-networks)
    - [Recurrent Neural Network-based Graph-level Learning](#recurrent-neural-network-based-graph-level-learning)
    - [Convolution Neural Network-based Graph-level Learning](#convolution-neural-network-based-graph-level-learning)
    - [Capsule Neural Network-based Graph-level Learning](#capsule-neural-network-based-graph-level-learning)
  - [Graph-Level Graph Neural Networks (GL-GNNs)](#graph-Level-graph-neural-networks)
    - [Message Passing Neural Networks](#message-passing-neural-networks)
    - [Subgraph-based GL-GNNs](#Subgraph-based-gl-gnns)
    - [Kernel-based GL-GNNs](#kernel-based-gl-gnns)
    - [Contrastive-based GL-GNNs](#contrastive-based-gl-gnns)
    - [Spectral-based GL-GNNs](#spectral-based-gl-gnns)
  - [Graph Pooling](#graph-pooling)
    - [Global Graph Pooling](#global-graph-pooling)
      - [Numeric Operation Pooling](#numeric-operation-pooling)
      - [Attention-based Pooling](#attention-based-pooling)
      - [Convolution Neural Network-based Pooling](#convolution-neural-network-based-pooling) 
      - [Global Top-K Pooling](#global-top-k-pooling)
    - [Hierarchical Graph Pooling](#hierarchical-graph-pooling)
      - [Clustering-based Pooling](#clustering-based-pooling)
      - [Hierarchical Top-K Pooling](#hierarchical-top-K-pooling)
      - [Hierarchical Tree-based Pooling](#hierarchical-tree-based-pooling)
  - [Datasets](#datasets)
  - [Applications](#Applications)
  - [Tools](#tools)  

----------
## A Timeline of Graph-level Learning
![timeline](timeline.png)
----------
## Surveys
| Paper Title | Venue | Year | Materials | 
| ---- | :----: | :----: | :----: | 
| A Comprehensive Survey of Graph-level Learning | arXiv | 2023 | [[Paper]()]|
| Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities | arXiv | 2022 | [[Paper](https://arxiv.org/pdf/2204.07321.pdf)]|
| Graph-level Neural Networks: Current Progress and Future Directions | arXiv | 2022 | [[Paper](https://arxiv.org/pdf/2205.15555.pdf)]| 
| A Survey on Graph Kernels | Appl. Netw. Sci. | 2020 | [[Paper](https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0195-3?ref=https://githubhelp.com)] |
| Deep Learning on Graphs: A Survey | IEEE Trans. Knowl. Data Eng. | 2020 | [[Paper](https://ieeexplore.ieee.org/abstract/document/9039675)] |
| A Comprehensive Survey on Graph Neural Networks | IEEE Trans. Neural Netw. Learn. Syst. | 2020| [[Paper](https://ieeexplore.ieee.org/abstract/document/9046288)] |

----------
## Traditional Graph-level Learning
### Graph Kernels
#### Message Passing Kernels
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| A Persistent Weisfeiler-lehman Procedure for Graph Classification | ICML | 2019 | P-WL | [[Paper](http://proceedings.mlr.press/v97/rieck19a/rieck19a.pdf)] [[Code](https://github.com/BorgwardtLab/P-WL)]|
| Glocalized Weisfeiler-lehman Graph Kernels: Global-local Feature Maps of Graphs | ICDM | 2017 | Global-WL | [[Paper](https://ieeexplore.ieee.org/abstract/document/8215505)] [[Code](https://github.com/chrsmrrs/glocalwl)]|
| Propagation kernels: Efficient Graph Kernels from Propagated Information | Mach. Learn. | 2016 | PK | [[Paper](https://link.springer.com/article/10.1007/s10994-015-5517-9)] [[Code](https://github.com/marionmari/propagation_kernels)]|
| Weisfeiler-lehman Graph Kernels | J. Mach. Learn. Res. | 2011 | WL | [[Paper](https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf)] [[Code](https://github.com/BorgwardtLab/graph-kernels)]|
| A linear-time graph kernel | ICDM | 2009 | NHK | [[Paper](https://ieeexplore.ieee.org/abstract/document/5360243)] [[Code](https://github.com/ysig/GraKeL)]|

#### Shortest Path Kernels 
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Shortest-path Graph Kernels for Document Similarity | EMNLP | 2017 | SPK-DS | [[Paper](https://aclanthology.org/D17-1202.pdf)]|
| Shortest-path Kernels on Graphs | ICDM | 2005 | SPK | [[Paper](https://ieeexplore.ieee.org/abstract/document/1565664)] [[Code](https://github.com/ysig/GraKeL)]|

#### Random Walk Kernels 
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Graph Kernels | J. Mach. Learn. Res. | 2010 | SOMRWK | [[Paper](https://www.jmlr.org/papers/volume11/vishwanathan10a/vishwanathan10a.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/ysig/GraKeL)]|
| Extensions of Marginalized Graph Kernels | ICML | 2004 | ERWK | [[Paper](https://dl.acm.org/doi/abs/10.1145/1015330.1015446)] [[Code](https://github.com/jajupmochi/graphkit-learn)]|
| On Graph Kernels: Hardness Results and Efficient Alternatives | LNAI | 2003 | RWK | [[Paper](https://link.springer.com/chapter/10.1007/978-3-540-45167-9_11)] [[Code](https://github.com/jajupmochi/graphkit-learn)]|

#### Optimal Assignment Kernels
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Transitive Assignment Kernels for Structural Classification | SIMBAD | 2015 | TAK | [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-24261-3_12)]|
| Learning With Similarity Functions on Graphs Using Matchings of Geometric Embeddings | KDD | 2015 | GE-OAK | [[Paper](https://dl.acm.org/doi/abs/10.1145/2783258.2783341)]|
| Solving the Multi-way Matching Problem by Permutation Synchronization | NeurIPS | 2013 | PS-OAK | [[Paper](https://proceedings.neurips.cc/paper/2013/file/3df1d4b96d8976ff5986393e8767f5b2-Paper.pdf)] [[Code](https://github.com/zju-3dv/multiway)]|
| Optimal Assignment Kernels for Attributed Molecular Graphs | ICML | 2005 | OAK | [[Paper](https://dl.acm.org/doi/abs/10.1145/1102351.1102380)]|

#### Subgraph Kernels
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Subgraph Matching Kernels for Attributed Graphs | ICML | 2012 | SMK | [[Paper](https://icml.cc/Conferences/2012/papers/542.pdf)] [[Code](https://github.com/fapaul/GraphKernelBenchmark)]|
| Fast Neighborhood Subgraph Pairwise Distance Kernel | ICML | 2010 | NSPDK | [[Paper](https://icml.cc/Conferences/2010/papers/347.pdf)] [[Code](https://github.com/fabriziocosta/EDeN)]|
| Efficient Graphlet Kernels for Large Graph Comparison | AISTATS | 2009 | Graphlet | [[Paper](http://proceedings.mlr.press/v5/shervashidze09a/shervashidze09a.pdf)] [[Code](https://github.com/ysig/GraKeL)]|

### Subgraph Mining
#### Frequent Subgraph Mining
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| gspan: Graph-based Substructure Pattern Mining | ICDM | 2002 | gspan | [[Paper](https://ieeexplore.ieee.org/abstract/document/1184038)] [[Code](https://github.com/betterenvi/gSpan)]|
| Frequent Subgraph Discovery | ICDM | 2001 | FSG | [[Paper](https://ieeexplore.ieee.org/abstract/document/989534)] [[Code](https://github.com/NikhilGupta1997/Data-Mining-Algorithms)]|
| An Apriori-based Algorithmfor Mining Frequent Substructures from Graph Data | ECML-PKDD | 2000 | AGM | [[Paper](https://link.springer.com/chapter/10.1007/3-540-45372-5_2)] [[Code](https://github.com/Aditi-Singla/Data-Mining)]|

#### Discriminative Subgraph Mining
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Multi-graph-view Learning for Graph Classification | ICDM | 2014 | gCGVFL | [[Paper](https://ieeexplore.ieee.org/abstract/document/7023376)]|
| Positive and Unlabeled Learning for Graph Classification | ICDM | 2011 | gPU | [[Paper](https://ieeexplore.ieee.org/abstract/document/6137301)]|
| Semi-supervised Feature Selection for Graph Classification | KDD | 2010 | gSSC | [[Paper](https://dl.acm.org/doi/abs/10.1145/1835804.1835905)]|
| Multi-label Feature Selection for Graph Classification | ICDM | 2010 | gMLC | [[Paper](https://ieeexplore.ieee.org/abstract/document/5693981)]|
| Near-optimal Supervised Feature Selection Among Frequent Subgraphs | SDM | 2009 | CORK | [[Paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611972795.92)]|
| Mining Significant Graph Patterns by Leap Search | SIGMOD | 2008 | LEAP | [[Paper](https://dl.acm.org/doi/abs/10.1145/1376616.1376662)]|

### Graph Embedding
#### Deterministic Graph Embedding
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Fast Attributed Graph Embedding via Density of States | ICDM | 2021 | A-DOGE | [[Paper](https://ieeexplore.ieee.org/abstract/document/9679053)] [[Code](https://github.com/sawlani/A-DOGE)]|
| Bridging the Gap Between Von Neumann Graph Entropy and Structural Information: Theory and Applications | WWW | 2021 | VNGE | [[Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449804)] [[Code](https://github.com/xuecheng27/WWW21-Structural-Information)]|
| Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs | WWW | 2021 | SLAQ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3366423.3380026)] [[Code](https://github.com/google-research/google-research/tree/master/graph_embedding/slaq)]|
| A Simple Yet Effective Baseline for Non-attributed Graph Classification | ICLR-RLGM | 2019 | LDP | [[Paper](https://arxiv.org/pdf/1811.03508.pdf)] [[Code](https://github.com/Chen-Cai-OSU/LDP)]|
| Anonymous Walk Embeddings | ICML | 2018 | AWE | [[Paper](http://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf)] [[Code](https://github.com/nd7141/AWE)]|
| Hunt For The Unique, Stable, Sparse And Fast Feature Learning On Graphs | NeurIPS | 2017 | FGSD | [[Paper](https://proceedings.neurips.cc/paper/2017/file/d2ddea18f00665ce8623e36bd4e3c7c5-Paper.pdf)] [[Code](https://github.com/vermaMachineLearning/FGSD)]|

#### Learnable Graph Embedding
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |
| Learning Graph Representation via Frequent Subgraphs | SDM | 2018 | GE-FSG | [[Paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611975321.35)] [[Code](https://github.com/nphdang/GE-FSG)]|
| graph2vec: Learning Distributed Representations of Graphs | KDD-MLG | 2017 | graph2vec | [[Paper](https://arxiv.org/pdf/1707.05005.pdf)] [[Code](https://github.com/MLDroid/graph2vec_tf)]|
| subgraph2vec: Learning Distributed Representations of Rooted Sub-graphs from Large Graphs | KDD-MLG | 2016 | subgraph2vec | [[Paper](https://arxiv.org/pdf/1606.08928.pdf)] [[Code](https://github.com/MLDroid/subgraph2vec_tf)]|

----------
## Graph-Level Deep Neural Networks
### Recurrent Neural Network-based Graph-level Learning
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models | ICML | 2018 | GraphRNN | [[Paper](http://proceedings.mlr.press/v80/you18a/you18a.pdf)] [[Code](https://github.com/snap-stanford/GraphRNN)]|
| NetGAN: Generating Graphs via Random Walks | ICML | 2018 | GraphRNN | [[Paper](http://proceedings.mlr.press/v80/bojchevski18a/bojchevski18a.pdf)] [[Code](https://github.com/danielzuegner/netgan)]|
| Substructure Assembling Network for Graph Classification | AAAI | 2018 | SAN | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/11742)]|
| Graph Classification using Structural Attention | KDD | 2018 | GAM | [[Paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3219980)] [[Code](https://github.com/benedekrozemberczki/GAM)]|
| Gated Graph Sequence Neural Networks | ICLR | 2016 | GGNN | [[Paper](https://arxiv.org/pdf/1511.05493.pdf)] [[Code](https://github.com/Microsoft/gated-graph-neural-network-samples)]|

### Convolution Neural Network-based Graph-level Learning
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Kernel Graph Convolutional Neural Networks | ICANN | 2018 | KCNN | [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-01418-6_3)] [[Code](https://github.com/giannisnik/cnn-graph-classification)]|
| Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs | CVPR | 2017 | ECC | [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.pdf)] [[Code](https://github.com/mys007/ecc)]|
| Diffusion-Convolutional Neural Networks | NeurIPS | 2016 | DCNN | [[Paper](https://proceedings.neurips.cc/paper/2016/file/390e982518a50e280d8e2b535462ec1f-Paper.pdf)] [[Code](https://github.com/jcatw/dcnn)]|
| Learning Convolutional Neural Networks for Graphs | ICML | 2016 | PATCHYSAN | [[Paper](http://proceedings.mlr.press/v48/niepert16.pdf)] [[Code](https://github.com/tvayer/PSCN)]|

### Capsule Neural Network-based Graph-level Learning
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
| Capsule Neural Networks for Graph Classification using Explicit Tensorial Graph Representations | arXiv | 2019 | PATCHYCaps | [[Paper](https://arxiv.org/pdf/1902.08399.pdf)] [[Code](https://github.com/BraintreeLtd/PatchyCapsules)]|
| Capsule Graph Neural Network | ICLR | 2019 | CapsGNN | [[Paper](https://openreview.net/pdf?id=Byl8BnRcYm)] [[Code](https://github.com/benedekrozemberczki/CapsGNN)]|
| Graph Capsule Convolutional Neural Networks | WCB | 2018 | GCAPSCNN | [[Paper](https://arxiv.org/pdf/1805.08090.pdf)] [[Code](https://github.com/vermaMachineLearning/Graph-Capsule-CNN-Networks)]|


----------
## Graph-Level Graph Neural Networks
### Message Passing Neural Networks
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |  
| The Surprising Power of Graph Neural Networks with Random Node Initialization | IJCAI | 2021RNI | [[Paper](https://arxiv.org/pdf/2010.01179.pdf)]|
| Weisfeiler and Lehman Go Cellular: CW Networks | NeurIPS | 2021 | CWN | [[Paper](https://proceedings.neurips.cc/paper/2021/file/157792e4abb490f99dbd738483e0d2d4-Paper.pdf)] [[Code](https://github.com/twitter-research/cwn)]|
| Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks | ICML | 2021 | SWL | [[Paper](http://proceedings.mlr.press/v139/bodnar21a/bodnar21a.pdf)] [[Code](https://github.com/twitter-research/cwn)]|
| Expressive Power of Invariant and Equivariant Graph Neural Networks | ICLR | 2021 | FGNN | [[Paper](https://arxiv.org/pdf/2006.15646.pdf)] [[Code](https://github.com/mlelarge/graph_neural_net)]|
| Relational Pooling for Graph Representations | ICML | 2019 | RP | [[Paper](http://proceedings.mlr.press/v97/murphy19a/murphy19a.pdf)] [[Code](https://github.com/PurdueMINDS/RelationalPooling)]|
| Provably Powerful Graph Networks | NeurIPS | 2019 | PPGN | [[Paper](https://proceedings.neurips.cc/paper/2019/file/bb04af0f7ecaee4aae62035497da1387-Paper.pdf)] [[Code](https://github.com/hadarser/ProvablyPowerfulGraphNetworks)]|
| Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks | AAAI | 2019 | K-GNN | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4384)] [[Code](https://github.com/chrsmrrs/k-gnn)]|
| How Powerful are Graph Neural Networks? | ICLR | 2019 | GIN | [[Paper](https://arxiv.org/pdf/1810.00826.pdf)] [[Code](https://github.com/weihua916/powerful-gnns)]|
| Quantum-chemical Insights from Deep Tensor Neural Networks | Nat. Commun. | 2017 | DTNN | [[Paper](https://www.nature.com/articles/ncomms13890)] [[Code](https://github.com/atomistic-machine-learning/dtnn)]|
| Neural Message Passing for Quantum Chemistry | ICML | 2017 | MPNN | [[Paper](http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)] [[Code](https://github.com/priba/nmp_qc)]|
| Interaction Networks for Learning about Objects, Relations and Physics | NeurIPS | 2016 | GraphSim | [[Paper](https://proceedings.neurips.cc/paper/2016/file/3147da8ab4a0437c15ef51a5cc7f2dc4-Paper.pdf)] [[Code](https://github.com/clvrai/Relation-Network-Tensorflow)]|
| Convolutional Networks on Graphs for Learning Molecular Fingerprints | NeurIPS | 2015 | Fingerprint | [[Paper](https://proceedings.neurips.cc/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf)] [[Code](https://github.com/HIPS/neural-fingerprint)]|



### Subgraph-based GL-GNNs
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |  


### Kernel-based GL-GNNs
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |  


### Contrastive-based GL-GNNs
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |  


### Spectral-based GL-GNNs
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: |  




----------
## Autoencoder-based Community Detection
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
|Community detection based on unsupervised attributed network embedding | Expert Syst. Appl. | 2023 | CDBNE | [[Paper](https://doi.org/10.1016/j.eswa.2022.118937)][[Code](https://github.com/xidizxc/CDBNE)] |
|Exploring temporal community structure via network embedding | IEEE TCYB | 2022 | VGRGMM | [[Paper](https://ieeexplore.ieee.org/abstract/document/9768181)]|
|Graph community infomax | ACM TKDD | 2022 | GCI | [[Paper](https://dl.acm.org/doi/10.1145/3480244)] |
|Multi-modal non-Euclidean brain network analysis with community detection and convolutional autoencoder | IEEE TETCI | 2022 | M2CDCA | [[Paper](https://ieeexplore.ieee.org/document/9773106)] |
|Deep neighbor-aware embedding for node clustering in attributed graphs | Pattern Recognit. | 2022 | DNENC | [[Paper](https://doi.org/10.1016/j.patcog.2021.108230)] |
|Semi-supervised overlapping community detection in attributed graph with graph convolutional autoencoder | Inf. Sci. | 2022 | SSGCAE | [[Paper](https://doi.org/10.1016/j.ins.2022.07.036)] |
|A weighted network community detection algorithm based on deep learning | Appl. Math. Comput. | 2021 | WCD | [[Paper](https://www.sciencedirect.com/science/article/pii/S0096300321000606)] |
| DNC: A deep neural network-based clustering-oriented network embedding algorithm | J. Netw. Comput. Appl. | 2021 | DNC | [[Paper](https://www.sciencedirect.com/science/article/pii/S1084804520303209)] |
|Boosting nonnegative matrix factorization based community detection with graph attention auto-encoder | IEEE TBD | 2021 | NMFGAAE | [[Paper](https://ieeexplore.ieee.org/abstract/document/9512416)]|
|Self-supervised graph convolutional network for multi-view clustering | IEEE TMM | 2021 | SGCMC | [[Paper](https://ieeexplore.ieee.org/document/9472979)] |
|Graph embedding clustering: Graph attention auto-encoder with cluster-specificity distribution | Neural Netw. | 2021 | GEC-CSD | [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608021002008)][[Code](https://github.com/xdweixia/SGCMC)] |
|An evolutionary autoencoder for dynamic community detection | Sci. China Inf. Sci. | 2020 | <nobr> sE-Autoencoder <nobr> | [[Paper](https://link.springer.com/article/10.1007/s11432-020-2827-9)] |
|Stacked autoencoder-based community detection method via an ensemble clustering framework | Inf. Sci. | 2020 | CDMEC | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S002002552030270X)] |
|Community-centric graph convolutional network for unsupervised community detection | IJCAI | 2020 | GUCD | [[Paper](https://www.ijcai.org/Proceedings/2020/0486.pdf)] |
|Structural deep clustering network |  WWW | 2020 | SDCN | [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380214)][[Code](https://github.com/bdy9527/SDCN)] |
|One2Multi graph autoencoder for multi-view graph clustering | WWW | 2020 | One2Multi | [[Paper](https://dl.acm.org/doi/10.1145/3366423.3380079)][[Code](https://github.com/googlebaba/WWW2020-O2MAC)] |
|Multi-view attribute graph convolution networks for clustering | IJCAI | 2020 | MAGCN | [[Paper](https://www.ijcai.org/Proceedings/2020/0411.pdf)] |
|Deep multi-graph clustering via attentive cross-graph association | WSDM | 2020 | DMGC | [[Paper](https://dl.acm.org/doi/abs/10.1145/3336191.3371806)][[Code](https://github.com/flyingdoog/DMGC)] |
|Effective decoding in graph auto-encoder using triadic closure | AAAI | 2020 | TGA <br> TVGA <br> | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5437/5293)] |
|Graph representation learning via ladder gamma variational autoencoders | AAAI | 2020 | LGVG | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6013/5869)] |
|High-performance community detection in social networks using a deep transitive autoencoder | Inf. Sci. | 2019 | <nobr> Transfer-CDDTA <nobr> | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0020025519303251)] |
|Attributed graph clustering: A deep attentional embedding approach | IJCAI | 2019 | DAEGC | [[Paper](https://www.ijcai.org/Proceedings/2019/0509.pdf)] |
|Stochastic blockmodels meet graph neural networks | ICML | 2019 | DGLFRM | [[Paper](http://proceedings.mlr.press/v97/mehta19a/mehta19a.pdf)][[Code](https://github.com/nikhil-dce/SBM-meet-GNN)] |
|Variational graph embedding and clustering with laplacian eigenmaps | IJCAI | 2019 | VGECLE | [[Paper](https://www.ijcai.org/Proceedings/2019/0297.pdf)] |
|Optimizing variational graph autoencoder for community detection | BigData | 2019 | VGAECD-OPT | [[Paper](https://ieeexplore.ieee.org/abstract/document/9006123)] |
|Integrative network embedding via deep joint reconstruction | IJCAI | 2018 | UWMNE | [[Paper](https://www.ijcai.org/Proceedings/2018/0473.pdf)] |
|Deep attributed network embedding | IJCAI | 2018 | DANE | [[Paper](https://www.ijcai.org/Proceedings/2018/0467.pdf)][[Code](https://github.com/gaoghc/DANE)] |
|Deep network embedding for graph representation learning in signed networks | IEEE TCYB | 2018 | DNE-SBP | [[Paper](https://ieeexplore.ieee.org/document/8486671)][[Code](https://github.com/shenxiaocam/Deep-network-embedding-for-graph-representation-learning-in-signed-networks)] |
|DFuzzy: A deep learning-based fuzzy clustering model for large graphs | Knowl. Inf.  Syst. | 2018 | DFuzzy | [[Paper](https://link.springer.com/article/10.1007/s10115-018-1156-3)] |
|Learning community structure with variational autoencoder | ICDM | 2018 | VGAECD | [[Paper](https://ieeexplore.ieee.org/document/8594831)] |
|Adversarially regularized graph autoencoder for graph embedding | IJCAI | 2018 | ARGA <br> ARVGA <br> | [[Paper](https://www.ijcai.org/Proceedings/2018/0362.pdf)][[Code](https://github.com/Ruiqi-Hu/ARGA)]| 
|BL-MNE: Emerging heterogeneous social network embedding through broad learning with aligned autoencoder | ICDM | 2017 | DIME | [[Paper](https://doi.org/10.1109/ICDM.2017.70)][[Code](http://www.ifmlab.org/files/code/Aligned-Autoencoder.zip)] |
|MGAE: Marginalized graph autoencoder for graph clustering | CIKM | 2017 | MGAE | [[Paper](https://dl.acm.org/doi/10.1145/3132847.3132967)][[Code](https://github.com/FakeTibbers/MGAE)] |
|Graph clustering with dynamic embedding | Preprint | 2017 | GRACE | [[Paper](https://arxiv.org/abs/1712.08249)] | 
|Modularity based community detection with deep learning | IJCAI | 2016 | semi-DRN | [[Paper](https://www.ijcai.org/Proceedings/16/Papers/321.pdf)][[Code](http://yangliang.github.io/code/DC.zip)] |
|Deep neural networks for learning graph representations | AAAI | 2016 | DNGR | [[Paper](https://dl.acm.org/doi/10.5555/3015812.3015982)] |
|Learning deep representations for graph clustering | AAAI | 2014 | GraphEncoder | [[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8527/8571)][[Code](https://github.com/quinngroup/deep-representations-clustering)] |

----------
## Other Deep Learning-based Community Detection 
| Paper Title | Venue | Year | Method | Materials | 
| ---- | :----: | :----: | :----: | :----: | 
|Deep alternating non-negative matrix factorisation | Knowl.-Based Syst. | 2022 |  DA-NMF | [[Paper](https://doi.org/10.1016/j.knosys.2022.109210)] |
|CGC: Contrastive Graph Clustering for Community Detection and Tracking | WWW | 2022 | CGC | [[Paper](https://dl.acm.org/doi/10.1145/3485447.3512160)] |
|Deep graph clustering via dual correlation reduction | AAAI | 2022 | DCRN | [[Paper](https://www.aaai.org/AAAI22Papers/AAAI-5928.LiuY.pdf)] [[Code](https://github.com/yueliu1999/DCRN)] |
|Cluster-aware heterogeneous information network embedding | WSDM | 2022 | VaCA-HINE | [[Paper](https://dl.acm.org/doi/abs/10.1145/3488560.3498385)] |
|Graph filter-based multi-view attributed graph clustering | IJCAI | 2021 | MvAGC | [[Paper](https://www.ijcai.org/proceedings/2021/0375.pdf)] [[Code](https://github.com/sckangz/MvAGC)] |
|A deep learning framework for self-evolving hierarchical community detection | CIKM | 2021 | ReinCom | [[Paper](https://dl.acm.org/doi/10.1145/3459637.3482223)] |
|Unsupervised learning of joint embeddings for node representation and community detection | ECML-PKDD | 2021 | J-ENC | [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-86520-7_2)] |
|Community detection based on modularized deep nonnegative matrix factorization | Int. J. Pattern Recognit. Artif. Intell. | 2020 | MDNMF | [[Paper](https://www.worldscientific.com/doi/abs/10.1142/S0218001421590060)] |
|Deep autoencoder-like nonnegative matrix factorization for community detection | CIKM | 2018 | DANMF | [[Paper](https://dl.acm.org/doi/10.1145/3269206.3271697)][[Code](https://github.com/benedekrozemberczki/DANMF)] |
|Community discovery in networks with deep sparse filtering | Pattern Recognit. | 2018 | DSFCD | [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132031830116X)] |
|A non-negative symmetric encoder-decoder approach for community detection | CIKM | 2017 | Sun _et al._ | [[Paper](https://dl.acm.org/doi/abs/10.1145/3132847.3132902)] |

----------
## Non-Deep Learning-based Communtiy Detection
| Paper Title | Venue | Year | Method | Materials |
| ---- | :----: | :----: | :----: | :----: |
|Community detection via autoencoder-like nonnegative tensor decomposition | IEEE TNNLS | 2022 | ANTD | [[Paper](https://ieeexplore.ieee.org/document/9904739)] |
|Graph regularized nonnegative matrix factorization for community detection in attributed networks| IEEE TNSE | 2022 | AGNMF-AN | [[Paper](https://ieeexplore.ieee.org/document/9904900)]
|Modeling and detecting communities in node attributed networks | IEEE TKDE | 2022 | CRSBM | [[Paper](https://ieeexplore.ieee.org/document/9852668)] |
|The trade-off between topology and content in community detection: An adaptive encoder-decoder-based NMF approach | Expert Syst. Appl. | 2022 | ANMF | [[Paper](https://doi.org/10.1016/j.eswa.2022.118230)] |
|Community detection in subspace of attribute | Inf. Sci. | 2022 | SOA | [[Paper](https://doi.org/10.1016/j.ins.2022.04.047)] |
|Explainability in graph data science: Interpretability, replicability, and reproducibility of community detection | IEEE Signal Process. Mag. | 2022 | --| [[Paper](https://ieeexplore.ieee.org/abstract/document/9810084)] |
|Differentially private community detection for stochastic block models | ICML | 2022 | Seif _et al._ | [[Paper](http://128.84.4.18/abs/2202.00636)] |
|Community detection in multiplex networks based on evolutionary multi-task optimization and evolutionary clustering ensemble | IEEE TEVC | 2022 | BSMCD | [[Paper](https://ieeexplore.ieee.org/document/9802693)] |
|Fine-grained attributed graph clustering | SDM | 2022 | FGC | [[Paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611977172.42)] [[Code](https://github.com/sckangz/FGC)] |
|HB-DSBM: Modeling the dynamic complex networks from community level to node level | IEEE TNNLS | 2022 | HB-DSBM | [[Paper](https://ieeexplore.ieee.org/document/9721420)]|
|PMCDM: Privacy-preserving multiresolution community detection in multiplex networks | Knowl.-Based Syst. | 2022 | PMCDM | [[Paper](https://doi.org/10.1016/j.knosys.2022.108542)] |
|Rearranging 'indivisible' blocks for community detection | IEEE TKDE | 2022 | RaidB | [[Paper](https://ieeexplore.ieee.org/document/9771068)] |
|Information diffusion-aware likelihood maximization optimization for community detection | Inf. Sci. | 2022 | EM-CD <br> L-Louvain <br> | [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025522003334)] |
|Community detection in partially observable social networks | ACM TKDD | 2022 | KroMFac | [[Paper](https://dl.acm.org/doi/abs/10.1145/3461339)] |
|Diverse and experienced group discovery via hypergraph clustering | SDM | 2022 | Amburg _et al._ | [[Paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611977172.17)] [[Code](https://github.com/ilyaamburg/fair-clustering-for-diverse-and-experienced-groups)] |
|Community detection in graph: An embedding method | IEEE TNSE | 2022 | SENMF | [[Paper](https://ieeexplore.ieee.org/abstract/document/9626627)] | 
|Community detection using local group assimilation | Expert Syst. Appl. | 2022| LGA | [[Paper](https://www.sciencedirect.com/science/article/pii/S0957417422010600)] |
|Identifying Early Warning Signals from News Using Network Community Detection | AAAI | 2022| Le Vine _et al._ | [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/21503/21252)] |
|Residual2Vec: Debiasing graph embedding with random graphs | NIPS | 2021 | residual2vec | [[Paper](https://proceedings.neurips.cc/paper/2021/file/ca9541826e97c4530b07dda2eba0e013-Paper.pdf)] [[Code](https://github.com/skojaku/residual2vec)] |
|Streaming belief propagation for community detection | NIPS | 2021 | StSBM | [[Paper](https://proceedings.neurips.cc/paper/2021/file/e2a2dcc36a08a345332c751b2f2e476c-Paper.pdf)] |
|Triangle-aware spectral sparsifiers and community detection | KDD | 2021 | Sotiropoulos _et al._ | [[Paper](https://dl.acm.org/doi/10.1145/3447548.3467260)] [[Code](https://www.dropbox.com/s/0p0ybkpx19jt3ii/codeKDDTriangleAware.zip?dl=0)] |
|Self-guided community detection on networks with missing edges | IJCAI | 2021 | SGCD | [[Paper](https://www.ijcai.org/proceedings/2021/0483.pdf)] |
|Effective and scalable clustering on massive attributed graphs | WWW | 2021 | ACMin | [[Paper](https://dl.acm.org/doi/10.1145/3442381.3449875)] [[Code](https://github.com/AnryYang/ACMin)] |
|Scalable Community Detection via Parallel Correlation Clustering | VLDB | 2021 | Shi _et al._ | [[Paper](http://vldb.org/pvldb/vol14/p2305-shi.pdf)] [[Code](https://github.com/jeshi96/parallel-correlation-clustering)] |
|Proximity-based group formation game model for community detection in social network | Knowl.-Based Syst. | 2021 | PBCD | [[Paper](https://linkinghub.elsevier.com/retrieve/pii/S0950705120307991)] |
|When random initializations help: A study of variational inference for community detection | J. Mach. Learn. Res. | 2021 | BCAVI | [[Paper](https://www.jmlr.org/papers/volume22/19-630/19-630.pdf)] |
|Compactness preserving community computation via a network generative process | IEEE TETCI | 2021 | FCOCD | [[Paper](https://ieeexplore.ieee.org/document/9548676)] |
|Identification of communities with multi-semantics via bayesian generative model | IEEE TBD | 2021 | ICMS | [[Paper](https://ieeexplore.ieee.org/document/9632396)] |
|A network embedding-enhanced Bayesian model for generalized community detection in complex networks | Inf. Sci. | 2021 | NEGCD | [[Paper](https://doi.org/10.1016/j.ins.2021.06.020)] |
|Multi-objective evolutionary clustering for large-scale dynamic community detection | Inf. Sci. | 2021 | <nobr> DYN-MODPSO <nobr> | [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025520311117)] |
|A joint community detection model: Integrating directed and undirected probabilistic graphical models via factor graph with attention mechanism | IEEE TBD | 2021 | AdaMRF | [[Paper](https://ieeexplore.ieee.org/document/9511816)] |
|Evolutionary markov dynamics for network community detection | IEEE TKDE | 2020 | ePMCL | [[Paper](https://ieeexplore.ieee.org/document/9099469)] |
|A network reduction-based multiobjective evolutionary algorithm for community detection in large-scale complex networks | IEEE TCYB | 2020 | RMOEA | [[Paper](https://ieeexplore.ieee.org/document/8486719)] |
|Integrating group homophily and individual personality of topics can better model network communities | ICDM | 2020 | GHIPT | [[Paper](https://ieeexplore.ieee.org/document/9338379)] |
|Community preserving network embedding based on memetic algorithm | IEEE TETCI | 2020 | MemeRep | [[Paper](https://ieeexplore.ieee.org/document/8449095)] |
|Detecting the evolving community structure in dynamic social networks | World Wide Web J. | 2020 | DECS | [[Paper](https://link.springer.com/article/10.1007/s11280-019-00710-z)] [[Code](https://github.com/FanzhenLiu/DECS)] |
|EdMot: An edge enhancement approach for motif-aware community detection | KDD | 2019 | EdMot | [[Paper](https://dl.acm.org/doi/10.1145/3292500.3330882)] |
|LPANNI: Overlapping community detection using label propagation in large-scale complex networks | IEEE TKDE | 2019 | LPANNI | [[Paper](https://ieeexplore.ieee.org/document/8443129)] |
|Detecting prosumer-community groups in smart grids from the multiagent perspective | IEEE TSMC | 2019 | PVMAS | [[Paper](https://ieeexplore.ieee.org/document/8660684)] |
|Local community mining on distributed and dynamic networks from a multiagent perspective | IEEE TCYB | 2016 | AOCCM | [[Paper](https://ieeexplore.ieee.org/document/7124425)] |
|General optimization technique for high-quality community detection in complex networks | Phys. Rev. E | 2014 | Combo | [[Paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.012811)] |
|Spectral methods for community detection and graph partitioning | Phys. Rev. E | 2013 | -- | [[Paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.88.042822)] |
|Stochastic blockmodels and community structure in networks | Phys. Rev. E | 2011 | DCSBM | [[Paper](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.016107)] |

----------
## Datasets
### Citation/Co-authorship Networks
- Citeseer, Cora, Pubmed https://linqs.soe.ucsc.edu/data
- DBLP http://snap.stanford.edu/data/com-DBLP.html
- Chemistry, Computer Science, Medicine, Engineering http://kddcup2016.azurewebsites.net/
### Online Social Networks
- Facebook http://snap.stanford.edu/data/ego-Facebook.html
- Epinions http://www.epinions.com/
- Youtube http://snap.stanford.edu/data/com-Youtube.html
- Last.fm https://www.last.fm/
- LiveJournal http://snap.stanford.edu/data/soc-LiveJournal1.html
- Gplus http://snap.stanford.edu/data/ego-Gplus.html
### Traditional Social Networks
- Cellphone Calls http://www.cs.umd.edu/hcil/VASTchallenge08/
- Enron Mail http://www.cs.cmu.edu/~enron/
- Friendship https://dl.acm.org/doi/10.1145/2501654.2501657
- Rados http://networkrepository.com/ia-radoslaw-email.php 
- Karate, Football, Dolphin http://www-personal.umich.edu/~mejn/netdata/
### Webpage Networks
- IMDb https://www.imdb.com/
- Wiki https://linqs.soe.ucsc.edu/data
### Product Co-purchasing Networks
- Amazon http://snap.stanford.edu/data/#amazon
### Other Networks
- Internet http://www-personal.umich.edu/~mejn/netdata/
- Java https://github.com/gephi/gephi/wiki/Datasets
- Hypertext http://www.sociopatterns.org/datasets
 
 ----------
## Tools
- Gephi https://gephi.org/
- Pajek http://mrvar.fdv.uni-lj.si/pajek/
- LFR https://www.santofortunato.net/resources

----------
**Disclaimer**

If you have any questions, please feel free to contact us.
Emails: <u>fanzhen.liu@hdr.mq.edu.au</u>, <u>xing.su2@hdr.mq.edu.au</u>
