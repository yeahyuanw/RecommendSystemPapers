# Must Read Papers on Recommend System

## Recall(召回)

- [2019-RecSys-Google: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)

  该论文提出了一个双塔模型用于Youtube的召回。传统的softmax在工业级应用中，计算量会非常大，所以普遍会采用基于采样的softmax。该论文采用了batch softmax，并考虑了采样带来的偏差（流式数据中，高频的item会被经常的采样到batch中）。论文通过计算item在流式数据中出现的平均间隔来计算item的概率，通过将item的概率应用到batch softmax的交叉熵loss中，来减少由于采样带来的偏差。

## Rank(排序)

- 2016-DLRS-Google: [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)

  该论文提出了Wide & Deep 模型来进行CTR预估。Wide & Deep模型通过结合线性模型的记忆能力和DNN模型的高阶特征交叉带来的泛化能力，来提升推荐业务效果。

- 2018-KDD-Alibaba: [Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/pdf/10.1145/3219819.3219823)

  该论文提出了DIN模型来进行CTR预估。DIN模型引入了attention机制，论文提出了Activation Weight，来计算候选Item与用户的历史行为序列的item的权重，然后对所有历史行为序列Item的embedding进行加权求和。

- 2019-AAAI-Alibaba: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://aimagazine.org/ojs/index.php/AAAI/article/view/4545)

  该论文提出了DIEN模型来进行CTR预估。DIEN包括了embedding层、Interest Extractor Layer、Interest Evolving Layer，最后将行为序列的向量、ad、user profile、context的向量进行拼接，输入到MLP进行预测。模型的核心模块就是：Interest Extractor Layer和Interest Evolving Layer

  Interest Extractor Layer: 用GRU来对用户点击行为序列之间的依赖进行建模。论文认为GRU的最终隐藏状态$h_T$只能表示最终兴趣导致了点击行为，并没有很好的利用到GRU的中间隐藏单元$h_t(t<T)$。论文认为兴趣会导致连续的多个点击行为，所以引入了辅助loss，用行为$b_{t+1}$来指导$h_t$的学习。

  Interest Evolving Layer: 对与target Ad相关的兴趣演化轨迹进行建模。论文提出了AUGRU--带注意力更新门的GRU结构。通过使用兴趣抽取层的GRU隐藏状态$h_t$和target Ad计算得到相关性$a_t$，再将GRU中更新门$u_t$乘以$a_t$。AUGRU可以减弱兴趣“漂移”带来的影响。

## Rerank(重排)

- 2018-NIPS-Hulu: [Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity](http://papers.nips.cc/paper/7805-fast-greedy-map-inference-for-determinantal-point-process-to-improve-recommendation-diversity.pdf)

  该论文提出用DPP(行列式点过程)算法来解决推荐中的多样性问题。

## MultiTask(多任务)

## ColdBoot(冷启动)

