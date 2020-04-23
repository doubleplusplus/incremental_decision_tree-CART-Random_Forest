# incremental-decision-tree-learner

Definition from wikipedia: [incremental decision tree](https://en.wikipedia.org/wiki/Incremental_decision_tree)

"An incremental decision tree algorithm is an online machine learning algorithm that outputs a decision tree. Many decision tree methods, construct a tree using a complete (static) dataset. Incremental decision tree methods allow an existing tree to be updated using only new data instances, without having to re-process past instances. This may be useful in situations where the entire dataset is not available when the tree is updated (i.e. the data was not stored), the original data set is too large to process or the characteristics of the data change over time (concept drift)."

## VFDT
This implementation is CART tree, based on the Hoeffding Tree i.e. very fast decision tree (VFDT) which is describe by the paper "Mining High-Speed Data Streams" (Domingos &amp; Hulten, 2000). The code is tested on dataset downloaded from UCI data base.

## EFDT
 "Extremely Fast Decision Tree" by Manapragada, Webb & Salehi (2018). As new data instances come in, EFDT can dynamically modify existing model, re-evaluate previous split or kill subtree. Now EFDT is available. But it runs slower than VFDT.

# Random Forest
Added implementation of Random Forest: `rf.py`. It is very efficient, because I used vectorized computation for computing gini impurity/index, and pooling for multi-processing.
