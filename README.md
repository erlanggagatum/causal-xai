# causal-xai

Submission code titled Causally Explainable AI with Bayesian Network and Sparse Autoencoder for Deep Learning Models.
The main code of Bayesian network construction is located in ```src\bayesian-network\bn-learning-bnlearn-monolithic-submit.ipynb``` and since the preprocessed dataset is available, this code can be run directly. 

#### Dataset download
Download the dataset in the link below, then extract it inside ```/src/```. The dataset folder should looks like ```/src/dataset/all_file_here.*```
```
https://drive.google.com/file/d/1s0NPFmGA3f77pJy-T4g7_LMkal8F4tDm/view?usp=sharing
```

#### Raw data preprocessing
```
src\preprocessing\preprocessing_pipeline_raw-submit.ipynb
```

#### DLP preprocessing (z)
```
src\preprocessing\preprocessing_pipeline_dlp-submit.ipynb
```

#### Preprocessing (Combine x, z, y_hat)
```
src\preprocessing\preprocessing_pipeline_result_combine-submit.ipynb
```

#### Sparse autoencoder
```
src\models\sparse-autoencoder-tf-submit.ipynb
```

#### Structure learning
```
src\bayesian-network\bn-learning-bnlearn-monolithic-submit.ipynb
```

#### Bayesian Network implementation in GeNIe

The GeNIe network file is located in ```src/model/bn-skeletion-final-submit```. This file only contains skeleton of the network. Use file from ```src/dataset/x_i_w_train_encoded.csv``` as the background dataset when doing the parameter learning in the software.
