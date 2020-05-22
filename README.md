# Must Read Papers on Recommend System

## Recall

- 2019-RecSys-Google: [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)

  该论文提出了一个双塔模型用于Youtube的召回。传统的softmax在工业级应用中，计算量会非常大，所以普遍会采用基于采样的softmax。该论文采用了batch softmax，并考虑了采样带来的偏差（流式数据中，高频的item会被经常的采样到batch中）。论文通过计算item在流式数据中出现的平均间隔来计算item的概率，通过将item的概率应用到batch softmax的交叉熵loss中，来减少由于采样带来的偏差。

## Rank

- 2014-KDD-Facebook: [Practical Lessons from Predicting Clicks on Ads at Facebook](https://dl.acm.org/doi/pdf/10.1145/2648584.2648589)

  Facebook提出了经典的GBDT+LR模型，利用GBDT进行特征筛选和组合，根据样本进入GBDT树的叶子节点，重新构建feature vector，输入到LR模型进行CTR的预测。为了评估CTR预测概率的精确性，还介绍了Normalized Entropy、Calibration两种评估方法。这篇论文还对LR模型的实时训练中的样本拼接，模型特征分析，负采样后的CTR校准等工程trick进行了介绍。

- 2016-DLRS-Google: [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)

  该论文提出了Wide & Deep 模型来进行CTR预估。Wide & Deep模型利用Wide部分结合线性模型的记忆能力、Deep部分的DNN模型为sparse feature学习到低维的dense embedding，对没有出现过的特征组合有更好的泛化性，同时能带来更高阶的非线性特征交叉。

- 2017-IJCAI-Huawei: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://www.ijcai.org/Proceedings/2017/0239.pdf)

  Wide & Deep模型的Wide部分的二阶特征还是需要人工组合。DeepFM模型利用FM模型来替换Wide & Deep中的Wide部分，自动进行二阶特征的组合。Deep部分和FM的二阶部分共用特征的embedding矩阵，学习高阶特征组合，提高特征交互的表征能力。

- 2018-KDD-Alibaba: [Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823)

  该论文提出了DIN模型来进行CTR预估。DIN模型引入了attention机制，论文提出了Activation Weight，来计算候选Item与用户的历史行为序列的item的权重，然后对所有历史行为序列Item的embedding进行加权求和。

- 2019-AAAI-Alibaba: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://aimagazine.org/ojs/index.php/AAAI/article/view/4545)

  该论文提出了DIEN模型来进行CTR预估。DIEN包括了embedding层、Interest Extractor Layer、Interest Evolving Layer，最后将行为序列的向量、ad、user profile、context的向量进行拼接，输入到MLP进行预测。模型的核心模块就是：Interest Extractor Layer和Interest Evolving Layer。

  Interest Extractor Layer: 用GRU来对用户点击行为序列之间的依赖进行建模。论文认为GRU的最终隐藏状态$h_T$只能表示最终兴趣导致了点击行为，并没有很好的利用到GRU的中间隐藏单元$h_t(t<T)$。论文认为兴趣会导致连续的多个点击行为，所以引入了辅助loss，用行为$b_{t+1}$来指导$h_t$的学习。

  Interest Evolving Layer: 对与target Ad相关的兴趣演化轨迹进行建模。论文提出了AUGRU--带注意力更新门的GRU结构。通过使用兴趣抽取层的GRU隐藏状态$h_t$和target Ad计算得到相关性$a_t$，再将GRU中更新门$u_t$乘以$a_t$。AUGRU可以减弱兴趣“漂移”带来的影响。

- 2018-SIGIR-Alibaba: [Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.07931)

  该论文提出了ESMM模型来进行CVR预估。论文认为传统的CVR预估存在两个问题：i) SSB：传统的CVR的训练集只是曝光样本中的点击样本，点击转化了的为正，点击未转化的为负。但是预测是的时候是所有曝光样本空间，训练数据和预测数据来自不同的分布，存在Sample Selection Bias问题；ii) CVR训练使用的点击样本远远小于CTR训练的曝光样本，所以存在Data Sparsity问题。

  ESMM是一个新的多任务框架，引入CTR和CTCVR作为辅助任务来学习CVR。ESMM的Loss函数由CTR和CTCVR两部分组成，并没有用到CVR任务的Loss。这样就可以在完整的曝光样本空间中进行训练和预测。$L(\theta_{cvr}, \theta_{ctr})=\sum^{N}_{i=1}l(y_i,f(x_i;\theta_{ctr})) + \sum^{N}_{i=1}l(y_i\&z_i,f(x_i;\theta_{ctr}*f(x_i;\theta_{cvr}))$

- 2019-CoRR-Meitu/Tencent: [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)

  该论文提出了FLEN模型来进行CTR预估。

## Rerank

- 1998-SIGIR-Carnegie Mellon: [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://dl.acm.org/doi/pdf/10.1145/3130348.3130369)

- 2018-NIPS-Hulu: [Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity](http://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-determinantal-point-process-to-improve-recommendation-diversity.pdf)

  该论文提出用DPP(行列式点过程)算法来解决推荐中的多样性问题。 通过预先定义好的相似度和多样性，在此基础上，通过DPP对相似度和多样性做一个权衡。具体理解可以参考博客：https://zhuanlan.zhihu.com/p/94464178

## MultiTask

## ColdBoot

### 1. Based on RL 

### 2. Based on GNN

## Calibration

- 2010-ICML-Microsoft: [Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search Advertising in Microsoft’s Bing Search Engine](https://icml.cc/Conferences/2010/papers/901.pdf)

- 2013-KDD-Google: [Ad Click Prediction: a View from the Trenches](https://dl.acm.org/doi/pdf/10.1145/2487575.2488200)

  