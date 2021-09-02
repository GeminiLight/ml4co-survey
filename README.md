# Papers on ML4CO

<div>
    <center style="color:#fc6423; font-size:24px; text-align:center;">Gemini Light<center>
	<center style="font-size:14px;"><a herf="mailto:wtfly2018@163.com">wtfly2018@163.com</a></center>
</div>
`Lasted Update: 2021/08/20`
`Current Version: v0.2 (P)`


## Preface

This is a brief survey on machine learning for combinatorial optimization, a fascinating and significant topic. We mainly collect relevant papers from top conferences, classify them into several categories by method types, and indicate additional information.

It's inevitable to have omissions or errors during the discussion for the limitation of the author's knowledge. Welcome to feedback them you found.

### To-Do List

- Continue to collect high-quality papers on ml4co.
- Add formal definitions and practical applications of every problem.
- Specify problem modelling in representative papers.
- Provide more detail for each model.
- Supply more analysis of development. 

### Combinatorial Optimization Problems

- **TSP**: Travelling Salesman Problem
- **VRP**: Vehicular Routing Problem
- **MAXCUT**: Maximum Cut
- **MVC**: Minimum Vertex Cover
- **MIS**: Maximal Independent Set
- **MC**: Maximal Clique
- **MCP**: Maximum Coverage Problem
- **SAT**: Satisfiability
- **GC**: Graph Coloring
- **JSSP**: Job Shop Scheduling Problem
- **3D-BPP**: 3D Bin-Packing Problem
- **MIP/MILP**: Mixed Integer (Linear) Programming

### Categories of Machine Learning

- **SL**: Supervised Learning
- **UL**: Unsupervised Learning
- **RL**: Reinforcement Learning *(Including Imitation Learning, IL)*

### Components of Neural Network

- **RNN**: Recurrent Neural Network *(Including LSTM, GRU)*
- **CNN**: Convolutional Neural Network
- **GNN**: Graph Neural Network *(Including Graph Embedding, GE)*
- **Attention** *(Including Self-Attention)*
- **Transformer**

## [Content](#content)

<table>
<tr><td><a href="#survey">Survey</a></td></tr>
<tr><td><a href="#analysis">Analysis</a></a></td></tr>
<tr><td><a href="#end-to-end">End-to-end</a></td></tr>
  <tr><td>&emsp;&emsp;<a href="#rnn/attention-based">RNN/Attention-based</a></td></tr>
  <tr><td>&emsp;&emsp;<a href="#gnn-based">GNN-based</a></td></tr>
  <tr><td>&emsp;&emsp;<a href="#other">Other</a></td></tr>
<tr><td><a href="#local-search">Local Search</a></td></tr>
<tr><td><a href="#b&b-based">B&B-based</a></td></tr>
</table>

## [Survey](#content)


1. **On Learning and Branching: a Survey**
   - `Publication`: TOP 2017
   - `Keyword`: Branch
   - `Link`: [paper](http://cerc-datascience.polymtl.ca/wp-content/uploads/2017/04/CERC_DS4DM_2017_004-1.pdf)
1. **Boosting Combinatorial Problem Modeling with Machine Learning**
   - `Publication`: IJCAI 2018
   - `Keyword`: ML
   - `Link`: [arXiv](https://arxiv.org/abs/1807.05517)
1. **A Review of Combinatorial Optimization with Graph Neural Networks**
   - `Publication`: BigDIA 2019
   - `Keyword`: GNN
   - `Link`: [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8802843)
1. **Learning Graph Matching and Related Combinatorial Optimization Problems**
   - `Publication`: IJCAI 2020
   - `Keyword`: GNN, Graph Matching
   - `Link`: [paper](https://www.ijcai.org/proceedings/2020/0694.pdf)
1. **Learning Combinatorial Optimization on Graphs: A Survey with Applications to Networking**
   - `Publication`: IEEE ACCESS 2020
   - `Keyword`: GNN, Computer Network
   - `Link`: [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9125934)
1. **A Survey on Reinforcement Learning for Combinatorial Optimization**
   - `Publication`: arXiv 2020
   - `Keyword`: RL
   - `Link`: [arXiv](https://arxiv.org/abs/2008.12248v2)
1. **A Survey on Reinforcement Learning for Combinatorial Optimization**
   - `Publication`: arXiv 2020
   - `Keyword`: RL
   - `Link`: [arXiv](https://arxiv.org/abs/2008.12248v2)
1. **Reinforcement Learning for Combinatorial Optimization: A Survey**
   - `Publication`: arXiv 2020
   - `Keyword`: RL
   - `Link`: [arXiv](https://arxiv.org/abs/2003.03600)
1. **Graph Learning for Combinatorial Optimization: A Survey of State-of-the-Art**
   - `Publication`: Data Science and Engineering 2021
   - `Keyword`: GNN
   - `Link`: [arXiv](https://arxiv.org/abs/2008.12646)
1. **Combinatorial optimization and reasoning with graph neural networks**
   - `Publication`: arXiv 2021
   - `Keyword`: GNN
   - `Link`: [arXiv](https://arxiv.org/abs/2102.09544)

## [Analysis](#content)

1. **Learning to Branch**
   - `Publication`: ICML 2018
   - `Keyword`: Branch
   - `Link`: [arXiv](https://arxiv.org/abs/1803.10150)
1. **Approximation Ratios of Graph Neural Networks for Combinatorial Problems**
   - `Publication`: NeurIPS 2019
   - `Keyword`: GNN
   - `Link`: [arXiv](https://arxiv.org/abs/1905.10261)
1. **On Learning Paradigms for the Travelling Salesman Problem**
   - `Publication`: NeurIPS 2019 (Workshop)
   - `Keyword`: RL vs SL
   - `Link`: [arXiv](https://arxiv.org/abs/1910.07210)
1. **On the Difficulty of Generalizing Reinforcement Learning Framework for Combinatorial Optimization**
   - `Publication`: ICML 2021 (Workshop)
   - `Keyword`: GNN
   - `Link`: [arXiv](https://arxiv.org/abs/2108.03713)


## [End-to-end](#content)


### [RNN/Attention-based](#content)

1. **Pointer Networks**
   - `Publication`: NeurIPS 2015
   - `CO-problem`: Finding planar convex hulls, computing Delaunay triangulations, TSP
   - `ML-type`: SL
   - `Component`: LSTM, Seq2Seq, Attention
   - `Innovation`: Ptr-Net not only improve over sequence-to-sequence with input attention, but also allow us to generalize to <u>variable size output dictionaries</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1506.03134)
   - ![image-20210815124017776](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815124017776.png)
1. **Neural Combinatorial Optimization with Reinforcement Learning**
   - `Publication`: ICLR 2017
   - `CO-problem`: TSP
   - `ML-type`: RL (Actor-critic)
   - `Component`: LSTM, Seq2Seq, Attention
   - `Innovation`: This paper presents a framework to tackle combinatorial optimization problems using neural networks and <u>reinforcement learning</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1611.09940)
   - ![image-20210815131544615](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815131544615.png)
1. **Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems**
   - `Publication`: AAAI 2021
   - `CO-problem`: VRP, TSP
   - `ML-type`: RL (REINFORCE)
   - `Component`: Self-Attention
   - `Innovation`: This paper proposes a Multi-Decoder Attention Model (MDAM) to train <u>multiple diverse policies</u>, which effectively increases the chance of finding good solutions compared with existing methods that train only one policy. A <u>customized beam search strategy</u> is designed to fully exploit the diversity of MDAM.
   - `Link`: [arXiv](https://arxiv.org/abs/2012.10638)
   - ![image-20210816135657473](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816135657473.png)
1. **Learning Improvement Heuristics for Solving Routing Problems**
    - `Publication`: TNNLS 2021
    - `CO-problem`: TSP, VRP
    - `ML-type`: RL (n-step Actor-Critic)
    - `Component`: Self-Attention
    - `Innovation`: This paper proposes a <u>self-attention based deep architecture</u> as the policy network to guide the selection of next solution.
    - `Link`: [arXiv](https://arxiv.org/abs/1912.05784)
    - ![image-20210816171739141](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816171739141.png)
1. **The Transformer Network for the Traveling Salesman Problem**
    - `Publication`: arXiv 2021
    - `CO-problem`: TSP
    - `ML-type`: RL
    - `Component`: Transformer
    - `Innovation`: This paper proposes to adapt the recent successful <u>Transformer architecture</u> originally developed for natural language processing to the combinatorial TSP.
    - `Link`: [arXiv](https://arxiv.org/abs/2103.03012)
    - ![image-20210816161717118](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816161717118.png)
1. **Matrix Encoding Networks for Neural Combinatorial Optimization**
   - `Publication`: arXiv 2021
   - `CO-problem`: TSP
   - `ML-type`: RL (REINFORCE)
   - `Component`: Attention
   - `Innovation`: MatNet is capable of <u>encoding matrix-style relationship data</u> found in many CO problems
   - `Link`: [arXiv](https://arxiv.org/abs/2106.11113)
   - ![image-20210816173143328](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816173143328.png)

### [GNN-based](#content)

1. **Learning Combinatorial Optimization Algorithms over Graphs**
   - `Publication`: NeurIPS 2017
   - `CO-problem`: MVC, MAXCUT, TSP
   - `ML-type`: RL (Q-learning)
   - `Component`: GNN (structure2vec, S2V)
   - `Innovation`:  This paper proposes a unique combination of reinforcement learning and <u>graph embedding</u> to address this challenge. The learned greedy policy behaves like a meta-algorithm that incrementally constructs a solution, and the action is determined by the output of a graph embedding network capturing the current state of the solution.
   - `Link`: [arXiv](https://arxiv.org/abs/1704.01665)
   - ![image-20210815122727875](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815122727875.png)
1. **Reinforcement Learning for Solving the Vehicle Routing Problem**
   - `Publication`: NeurIPS 2018
   - `CO-problem`: VRP
   - `ML-type`: RL
   - `Component`: GNN (GCN)
   - `Innovation`: This paper presents an end-to-end framework for solving the <u>Vehicle Routing Problem (VRP)</u> using reinforcement learning.
   - `Link`: [arXiv](https://arxiv.org/abs/1802.04240)
   - ![image-20210815125454294](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815125454294.png)
1. **Attention, Learn to Solve Routing Problems!**
   - `Publication`: NeurIPS 2018
   - `CO-problem`: TSP, VRP
   - `ML-type`: RL (REINFORCE+rollout baseline)
   - `Component`: Attention, GNN
   - `Innovation`: This paper proposes a model based on <u>attention layers</u> with benefits over the Pointer Network and we show how to train this model using REINFORCE with a simple baseline based on a deterministic greedy rollout, which we find is more efficient than using a value function.
   - `Link`: [arXiv](https://arxiv.org/abs/1803.08475)
   - ![image-20210815133136779](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815133136779.png)
1. **Learning to Solve NP-Complete Problems - A Graph Neural Network for Decision TSP**
   - `Publication`: AAAI 2019
   - `CO-problem`: TSP
   - `ML-type`: SL
   - `Component`: RNN, GNN
   - `Innovation`: This paper proposes <u>a GNN model where edges (embedded with their weights) communicate with vertices</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1809.02721)
   - ![image-20210815133954824](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815133954824.png)
1. **Learning a SAT Solver from Single-Bit Supervision**
   - `Publication`: ICLR 2019
   - `CO-problem`: SAT
   - `ML-type`: SL
   - `Component`: GNN, LSTM
   - `Innovation`: NeuroSAT enforces both <u>permutation invariance and negation invariance</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1802.03685)
   - ![image-20210815135231623](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815135231623.png)
   - ![image-20210815135755778](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815135755778.png)
1. **End to end learning and optimization on graphs**
   - `Publication`: NeurIPS 2019
   - `CO-problem`: MAXCUT
   - `ML-type`: UL
   - `Component`: GNN
   - `Innovation`: This paper proposed a new approach CLUSTERNET  to this <u>decision-focused learning</u> problem: include <u>a differentiable solver for a simple proxy to the true</u>, difficult optimization problem and learn a representation that maps the difficult problem to the simpler one. (relax+differentiate)
   - `Link`: [arXiv](https://arxiv.org/abs/1905.13732)
   - ![image-20210815143420270](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815143420270.png)
1. **Efficiently Solving the Practical Vehicle Routing Problem: A Novel Joint Learning Approach**
   - `Publication`: KDD 2020
   - `CO-problem`: VRP
   - `ML-type`: SL, RL (REINFORCE+rollout baseline)
   - `Component`: GCN
   - `Innovation`: GCN-NPEC model is based on the graph convolutional network (GCN) with node feature (coordination and demand) and edge feature (the real distance between nodes) as input and embedded. <u>Separate decoders</u> are proposed to decode the representations of these two features. The output of one decoder is the supervision of the other decoder. This paper also proposes a strategy that <u>combines the reinforcement learning manner with the supervised learning manner</u> to train the model.
   - `Link`: [paper](https://www.kdd.org/kdd2020/accepted-papers/view/efficiently-solving-the-practical-vehicle-routing-problem-a-novel-joint-lea)
   - ![image-20210816144127741](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816144127741.png)
1. **Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning**
   - `Publication`: NeurIPS 2020
   - `CO-problem`: JSSP
   - `ML-type`: RL (PPO)
   - `Component`: GNN
   - `Innovation`: This paper <u>exploit the disjunctive graph representation of JSSP</u> , and propose a Graph Neural Network based scheme to embed the states encountered during solving. 
   - `Link`: [arXiv](https://arxiv.org/abs/2010.12367)
   - ![image-20210816170033736](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816170033736.png)
1. **Learning to Solve Combinatorial Optimization Problems on Real-World Graphs in Linear Time**
   - `Publication`: arXiv 2020
   - `CO-problem`: TSP, VRP
   - `ML-type`: RL
   - `Component`: GNN
   - `Innovation`: This paper develops a new framework to <u>solve any combinatorial optimization problem over graphs that can be formulated as a single player game</u> defined by states, actions, and rewards, including minimum spanning tree, shortest paths, traveling salesman problem, and vehicle routing problem, problem, without expert knowledge.
   - `Link`: [arXiv](https://arxiv.org/abs/2006.03750)
   - ![image-20210816150049627](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816150049627.png)
1. **Combinatorial Optimization by Graph Pointer Networks and Hierarchical Reinforcement Learning**
   - `Publication`: AAAI 2020 (Workshop)
   - `CO-problem`: TSP
   - `ML-type`: Hierarchical RL
   - `Component`: GNN
   - `Innovation`: GPNs build upon <u>Pointer Networks</u> by introducing a <u>graph embedding layer</u> on the input, which captures relationships between nodes. Furthermore, to approximate solutions to constrained combinatorial optimization problems such as the TSP with time windows, we train <u>hierarchical GPNs (HGPNs) using RL</u>, which learns a hierarchical policy to find an optimal city permutation under constraints.
   - `Link`: [arXiv](https://arxiv.org/abs/1911.04936)
   - ![image-20210816171353265](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816171353265.png)
1. **A Bi-Level Framework for Learning to Solve Combinatorial Optimization on Graphs**
   - `Publication`: arXiv 2021
   - `CO-problem`: Directed Acyclic Graph scheduling, Graph Edit Distance, Hamiltonian Cycle Problem
   - `ML-type`: RL (PPO)
   - `Component`: GNN, ResNet, Attention
   - `Innovation`: This paper proposes a <u>bi-level framework</u> is developed with an upper-level learning method to optimize the graph (e.g. add, delete or modify edges in a graph), fused with a lower-level heuristic algorithm solving on the optimized graph. Such a bi-level approach <u>simplifies the learning on the original hard CO</u> and <u>can effectively mitigate the demand for model capacity</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/2106.04927)
   - ![image-20210816182855878](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816182855878.png)
1. **SOLO: Search Online, Learn Offline for Combinatorial Optimization Problems**
   - `Publication`: arXiv 2021
   - `CO-problem`: VRP, JSSP
   - `ML-type`: RL (DQN, MCTS)
   - `Component`: GNN
   - `Innovation`: Learn Offline -> DQN + Search Online -> MCTS
   - `Link`: [arXiv](https://arxiv.org/abs/2104.01646)
   - ![image-20210816163030736](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816163030736.png)

### [Other](#content)

1. **Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning**
   - `Publication`: NeurIPS 2020
   - `CO-problem`: VRP
   - `ML-type`: RL
   - `Component`: /
   - `Innovation`:  <u>Policy evaluation with mixed-integer optimization</u>. At the policy evaluation step, they formulate the action selection problem from each state as a mixed-integer program, in which they combine the combinatorial structure of the action space with the neural architecture of the value function by adapting the branch-and-cut approach.
   - `Link`: [arXiv](https://arxiv.org/abs/2010.12367)


## [Local Search](#content)

1. **Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search**
   - `Publication`: NeurIPS 2018
   - `CO-problem`: MIS, MVC, MC, SAT
   - `ML-type`: SL
   - `Component`: GNN (GCN)
   - `Innovation`: This paper proposes a model whose central component is a <u>graph convolutional network</u> that is trained to estimate the likelihood, for each vertex in a graph, of whether this vertex is part of the optimal solution. The network is designed and trained to <u>synthesize a diverse set of solutions</u>, which enables rapid exploration of the solution space via tree search.
   - `Link`: [arXiv](https://arxiv.org/abs/1704.01665)
   - ![image-20210815124857154](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210815124857154.png)
1. **Learning to Perform Local Rewriting for Combinatorial Optimization**
   - `Publication`: NeurIPS 2019
   - `CO-problem`: JSSP, VRP
   - `ML-type`: RL (Actor-Critic)
   - `Component`: LSTM
   - `Innovation`: NeuRewriter  learns a policy to pick heuristics, and <u>rewrite local components of the current solution to iteratively improve it until convergence</u>. The policy factorizes into a region-picking and a rule-picking component, each of which parameterized by an NN trained with actor-critic in RL.
   - `Link`: [arXiv](https://arxiv.org/abs/1810.00337)
   - ![image-20210816174005858](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816174005858.png)
1. **A Learning-based Iterative Method for Solving Vehicle Routing Problems**
   
   - `Publication`: ICLR 2020
   - `CO-problem`: VRP
   - `ML-type`: RL (REINFORCE)
   - `Component`: /
   - `Innovation`:  “Learn to Improve” (L2I)
     - hierarchical framework:  separate heuristic operators into two classes, namely <u>improvement operators and perturbation operators</u>. At each state, we choose the class first and then choose operators within the class. Learning from the current solution is made easier by focusing RL on the improvement operators only.
     - ensemble method: trains several RL policies at the same time, but with different state input features.
   - `Link`: [paper](https://openreview.net/forum?id=BJe1334YDH)
   - ![image-20210816174813049](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816174813049.png)
   
1. **Exploratory Combinatorial Optimization with Reinforcement Learning**
   
   - `Publication`: AAAI 2020
   - `CO-problem`: MAXCUT
   - `ML-type`: RL (DQN)
   - `Component`: GNN (MPNN)
   - `Innovation`: ECO-DQN combines <u>S2V-DQN, Reversible Actions, Observation Tuning and Intermediate Rewards</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1909.04063)
1. **Rewriting By Generating: Learn To Solve Large-Scale Vehicle Routing Problems**
   
   - `Publication`: ICLR 2021
   - `CO-problem`: VRP
   - `ML-type`: Hierarchical RL (REINFORCE)
   - `Component`: LSTM, K-means, PCA
   - `Innovation`: Inspired by the <u>classical idea of Divide-and-Conquer</u>, this paper presents a novel <u>Rewriting-by-Generating(RBG) framework with hierarchical RL agents</u> to solve large-scale VRPs. <u>RBG consists of a rewriter agent that refines the customer division globally and an elementary generator to infer regional solutions locally</u>.
   - `Link`: [paper](https://openreview.net/forum?id=xxWl2oEvP2h)
   - ![image-20210816175009718](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816175009718.png)
   
   ![image-20210816175223823](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816175223823.png)


## [B&B-based](#content)


1. **Learning to Search in Branch and Bound Algorithms**
   - `Publication`: NeurIPS 2014
   - `CO-problem`: MILP
   - `ML-type`: RL (Imitation learning)
   - `Component`: /
   - `Innovation`: This paper proposed an approach that <u>learns branch-and-bound by imitation learning</u>. (node selection policy and node pruning policy)
   - `Link`: [paper](https://papers.nips.cc/paper/5495-learning-to-search-in-branch-and-bound-algorithms)
   - ![image-20210816160633086](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816160633086.png)
1. **Learning to Branch in Mixed Integer Programming**
   - `Publication`: AAAI 2016
   - `CO-problem`: MIP
   - `ML-type`: RL (Imitation Learning)
   - `Component`: /
   - `Innovation`: This paper proposes <u>the first successful ML framework for variable selection in MIP</u>.
   - `Link`: [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPDFInterstitial/12514/11657)
1. **Learning to Run Heuristics in Tree Search**
   - `Publication`: IJCAI 2017
   - `CO-problem`: MIP
   - `ML-type`: SL
   - `Component`: classifier
   - `Innovation`: Central to this approach is the <u>use of Machine Learning (ML) for predicting whether a heuristic will succeed at a given node</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1803.10150)
1. **Improving Optimization Bounds using Machine Learning: Decision Diagrams meet Deep Reinforcement Learning**
   - `Publication`: AAAI 2019
   - `CO-problem`: MIS, MAXCUT
   - `ML-type`: RL (Q-learning)
   - `Component`: /
   - `Innovation`: This paper proposes an innovative and generic approach based on deep reinforcement learning for <u>obtaining an ordering for tightening the bounds</u> obtained with <u>relaxed and restricted DDs (decision diagrams)</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1809.03359)
   - ![image-20210816182155426](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816182155426.png)
1. **Exact Combinatorial Optimization with Graph Convolutional Neural Networks**
   - `Publication`: NeurIPS 2019
   - `CO-problem`: MIIP
   - `ML-type`: RL (Imitation learning)
   - `Component`: GNN (GCN)
   - `Innovation`: This paper proposes a <u>new graph convolutional neural network</u> model for learning branch-and-bound variable selection policies, which <u>leverages the natural variable-constraint bipartite graph representation of mixed-integer linear programs</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/1906.01629)
   - ![image-20210816134100224](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816134100224.png)
1. **Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization**
   - `Publication`: AAAI 2021
   - `CO-problem`: TSP, Portfolio Optimization Problem
   - `ML-type`: RL (DQN, PPO)
   - `Component`: GNN (GAT), Set Transformer
   - `Innovation`: This paper propose a general and hybrid approach, based on DRL and CP (Constraint Programming), for solving combinatorial optimization problems. The core of this approach is <u>based on a dynamic programming formulation, that acts as a bridge between both techniques</u>.
   - `Link`: [arXiv](https://arxiv.org/abs/2006.01610)
   - ![image-20210816155601272](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816155601272.png)
1. **Solving Mixed Integer Programs Using Neural Networks**
   - `Publication`: arXiv 2021
   - `CO-problem`: MIP
   - `ML-type`: RL (Imitation Learning)
   - `Component`: GNN (Bipartite graph)
   - `Innovation`: <u>Neural Diving + Neural Branching</u>
     - Neural Diving learns a deep neural network to generate multiple partial assignments for its integer variables, and the resulting smaller MIPs for un-assigned variables are solved with SCIP to construct high quality joint assignments.
     - Neural Branching learns a deep neural network to make variable selection decisions in branch-and-bound to bound the objective value gap with a small tree.
   - `Link`: [arXiv](https://arxiv.org/abs/2012.13349)
   - ![image-20210816161449504](C:\Users\Gemini向光性\AppData\Roaming\Typora\typora-user-images\image-20210816161449504.png)


## Appendix


### Template

1. ****
   - `Publication`: 
   - `CO-problem`: 
   - `ML-type`: 
   - `Component`: 
   - `Innovation`: 
   - `Link`: [arXiv]()
   -

1. ****
   - `Publication`: 
   - `CO-problem`: 
   - `Link`: [arXiv]()

### To Be Classified

1. **Reinforcement Learning for (Mixed) Integer Programming: Smart Feasibility Pump**
   - `Publication`: ICML 2021 (Workshop)
   - `CO-problem`: MIP
   - `Link`: [arXiv](https://arxiv.org/pdf/2102.09663.pdf)
1. **Learning Local Search Heuristics for Boolean Satisfiability**
   - `Publication`: NeurIPS 2019
   - `CO-problem`: SAT
   - `Link`: [paper](https://papers.nips.cc/paper/2019/hash/12e59a33dea1bf0630f46edfe13d6ea2-Abstract.html)
1. **Reinforcement Learning on Job Shop Scheduling Problems Using Graph Networks**
   - `Publication`: arXiv 2020
   - `CO-problem`: JSSP
   - `Link`: [arXiv](https://arxiv.org/abs/2009.03836)
1. **A Generalized Reinforcement Learning Algorithm for Online 3D Bin-Packing**
   - `Publication`: AAAI 2020 (Workshop)
   - `CO-problem`: 3D-BPP
   - `Link`: [arXiv](https://arxiv.org/abs/2007.00463)
1. **A Reinforcement Learning Approach to Job-shop Scheduling** 
   - `Publication`: IJCAI 2020
   - `CO-problem`: JSSP
   - `Link`: [paper](https://www.ijcai.org/Proceedings/95-2/Papers/013.pdf)
1. **A Reinforcement Learning Environment For Job-Shop Scheduling** 
   - `Publication`: arXiv 2021
   - `CO-problem`: JSSP
   - `Link`: [arXiv](https://arxiv.org/abs/2104.03760)
1. **Improving Optimization Bounds Using Machine Learning ~ Decision Diagrams Meet Deep Reinforcement Learning**
   - `Publication`: AAAI 2019
   - `CO-problem`: MIS, MAXCUT
   - `Link`: [arXiv](https://arxiv.org/abs/1809.03359)
1. **Online 3D Bin Packing with Constrained Deep Reinforcement Learning**
   - `Publication`: AAAI 2021
   - `CO-problem`: 3D-BPP
   - `Link`: [arXiv](https://arxiv.org/abs/2006.14978)
1. **A Data-Driven Approach for Multi-level Packing Problems in Manufacturing Industry**
   - `Publication`: KDD 2019
   - `CO-problem`: 3D-BPP
   - `Link`: [ACM DL](https://dl.acm.org/doi/10.1145/3292500.3330708)
1. **Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method**
   - `Publication`: arXiv 2017
   - `CO-problem`: 3D-BPP
   - `Link`: [arXiv](https://arxiv.org/abs/1708.05930)
1. **Meta-Learning-based Deep Reinforcement Learning for Multiobjective Optimization Problems**
   - `Publication`: arXiv 2017
   - `CO-problem`: TSP
   - `Link`: [arXiv](https://arxiv.org/abs/2105.02741)
1. **Dynamic Job-Shop Scheduling Using Reinforcement Learning Agents**
   - `Publication`: ROBOT AUTON SYST 2000
   - `CO-problem`: JSSP
   - `Link`: [paper](https://www.researchgate.net/publication/280532592_Dynamic_job_shop_scheduling_using_intelligent_agents)