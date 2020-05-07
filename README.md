# Must Read Papers on Recommend System

## Recall(召回)

- [2019-RecSys: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)

  该论文提出了一个双塔模型用于Youtube的召回。传统的softmax在工业级应用中，计算量会非常大，所以普遍会采用基于采样的softmax。该论文采用了batch softmax，并考虑了采样带来的偏差（流式数据中，高频的item会被经常的采样到batch中）。论文通过计算item在流式数据中出现的平均间隔来计算item的概率，通过将item的概率应用到batch softmax的交叉熵loss中，来减少由于采样带来的偏差。


## Rough Rank(粗排)

## Rank(精排)

## Rerank(重排)

## MultiTask(多任务)

## ColdBoot(冷启动)

