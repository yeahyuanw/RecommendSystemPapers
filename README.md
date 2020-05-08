# Must Read Papers on Recommend System

## Recall(召回)

- [2019-RecSys: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)

  该论文提出了一个双塔模型用于Youtube的召回。传统的softmax在工业级应用中，计算量会非常大，所以普遍会采用基于采样的softmax。该论文采用了batch softmax，并考虑了采样带来的偏差（流式数据中，高频的item会被经常的采样到batch中）。论文通过计算item在流式数据中出现的平均间隔来计算item的概率，通过将item的概率应用到batch softmax的交叉熵loss中，来减少由于采样带来的偏差。

## Rank(排序)

- 2019-AAAI: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://aimagazine.org/ojs/index.php/AAAI/article/view/4545)

  该论文提出了DIEN模型来进行CTR预估。DIEN包括了embedding层、Interest Extractor Layer、Interest Evolving Layer，最后将行为序列的向量、ad、user profile、context的向量进行拼接，输入到MLP进行预测。模型的核心模块就是：Interest Extractor Layer和Interest Evolving Layer

  - Interest Extractor Layer: 用GRU来对用户点击行为序列之间的依赖进行建模。论文认为GRU的最终隐藏状态$h_T$只能表示最终兴趣导致了点击行为，并没有很好的利用到GRU的中间隐藏单元$h_t(t<T)$。论文认为兴趣会导致连续的多个点击行为，所以引入了辅助loss，用行为$b_{t+1}$来指导$h_t$的学习。

## Rerank(重排)

## MultiTask(多任务)

## ColdBoot(冷启动)

