# 台灣ETF價格預測競賽
隊伍: NCU_newbie
成員：[陳廷睿](https://github.com/Ray941216/TBrainETF)、[劉亞昇](#)、[連丞宥](https://github.com/littlelienpeanut/ETF_prediction)、[曾翊銘](#)

## Abstract
本次比賽使用特徵除了從主辦方提供的資料集中計算KDJ技術指標、行為偵測指標、也將原始價格資料進行標準化、同時抓取三大法人交易資料標準化後當成特徵。
使用的演算法有 Multi-Task Learning, Convolutional-LSTM, Residual Convolutional-LSTM, Attention-Based LSTM, Simple LSTM, Ensemble Learning(Weighted Majority Voting).

模型訓練的方式根據模型分為只使用目標18檔的資料，以及使用全部資料的訓練。
