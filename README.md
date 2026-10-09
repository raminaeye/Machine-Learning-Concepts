# Machine Learning Concepts

Implementations of common machine learning algorithms in NumPy and PyTorch, written for learning. Each notebook builds one algorithm from scratch (or close to it), trains it on synthetic or public data, and visualizes what it learned.

Author: Ramin Anushiravani

## Notebooks

### Regression (`notebooks/regression/`)

* `Linear_Regression.ipynb`: linear regression with gradient descent and L2 regularization in NumPy, polynomial features, and the closed-form ridge solution
* `Linear and Non-Linear Regression Pytorch.ipynb`: linear regression in PyTorch, then why linear models fail on nonlinear data and how ReLU networks fix it
* `Logistic_Regression.ipynb`: multiclass logistic regression from scratch in NumPy on the Iris dataset
* `Logistic_Regression_Pytorch.ipynb`: logistic regression vs small neural networks on the two-moons dataset, with decision boundaries, ROC, and confusion matrices
* `Logistic_Regression_SimpleNN_Pytorch.ipynb`: overfitting on purpose with an oversized network, then fixing it with L1 regularization, L2 regularization, and dropout

### Trees (`notebooks/trees/`)

* `Decision_tree.ipynb`: entropy, information gain, and Gini impurity worked through by hand on Iris
* `decision_tree-random-forest-pytorch.ipynb`: decision tree and random forest from scratch in PyTorch
* `gradient_boosting-pytorch.ipynb`: gradient boosted trees from scratch in PyTorch

### Support Vector Machines (`notebooks/svm/`)

* `svm-pytorch.ipynb`: a neural network trained with the hinge loss (the SVM objective) on synthetic data

### Naive Bayes (`notebooks/naive-bayes/`)

* `Naive Bayes.ipynb`: multinomial naive Bayes with Laplace smoothing from scratch on the 20 Newsgroups dataset

### K-Nearest Neighbors (`notebooks/knn/`)

* `k-nearest-pytorch.ipynb`: k-NN classification in PyTorch with decision boundary plots showing the effect of k on overfitting

### K-Means (`notebooks/kmeans/`)

* `k-means_numpy.ipynb` and `k-means_pytorch.ipynb`: k-means clustering from scratch in NumPy and in PyTorch

### Dimensionality Reduction (`notebooks/dim-reduce/`)

* `pca.ipynb`: PCA on the Olivetti faces dataset with scikit-learn, then reimplemented from scratch in NumPy (eigenfaces, variance explained)
* `tsne.ipynb`: placeholder for a from-scratch t-SNE demo

### Neural Networks (`notebooks/neural/`)

* `backprop_pytorch.ipynb`: manual backpropagation for a small fully connected network, no autograd
* `RNN_pytorch.ipynb`: char-level LSTM language model trained on a text corpus, with text sampling after each epoch
* `CNN_pytorch.ipynb`: placeholder for a from-scratch CNN demo

### Deep Learning Helpers (`notebooks/dl-helpers/`)

* `BatchNorm.ipynb`: batch normalization, layer normalization, and dropout from scratch in NumPy, plus a minimal PyTorch training loop

### Applied Tasks (`notebooks/tasks/`)

* `Pytorch-Classification.ipynb`: text classification on AG News with torchtext (custom tokenizer, embedding bag classifier)
* `Recommendation.ipynb`: content-based, collaborative filtering, and hybrid recommenders on the MovieLens dataset

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open any notebook under `notebooks/` and run all cells.

Data notes:

* `data/movies.csv`, `data/ratings.csv`, and `data/training_text.txt` are committed in `data/` and used by the recommendation and RNN notebooks.
* Some notebooks download public datasets on first run: Olivetti faces and 20 Newsgroups (scikit-learn), AG News (torchtext). You need internet access for those cells.
* The `.png` files in `data/` are explanatory diagrams shown at the top of several notebooks.

Compute notes: most notebooks run in seconds to a few minutes on CPU. The RNN language model, the gradient boosting demo (100 trees), and the manual backprop demo (150k epochs) take noticeably longer.

## What you will learn

Each notebook follows the same pattern: build the algorithm from its math, train it, and look at the result. Highlights:

* How gradient descent, backpropagation, and regularization actually work, by implementing them by hand
* Why linear models fail on nonlinear data and what hidden layers change
* How trees split (entropy, Gini), and how forests and boosting combine them
* How embeddings turn text and user/item ids into something a network can learn from

## Disclaimer

Explanatory definitions and diagrams at the top of some notebooks were drafted with ChatGPT and the code was iterated from there. Everything here is for educational purposes.
