# Music Year Predictor

> This work has been completed by Peter Huang, Ryan Kim and Horatiu Lazu

## Objective
The purpose of this project is to analyze and evaluate the performance of classical machine learning methods (decision trees, support vector regression, random forests, linear regression) along with Convolutional Neural Networks (CNNs) for predicting the trending year of a given song. We take existing research and open-source implementations for music genre classifiation, and by leveraging the techniques applied there we adapt and attempt to solve this different type of problem.

This repository includes the implementation and scripts used for reproducing the results outlined in the paper.

## Getting Started
There are four components to this project.

1. Training + Validation Dataset
2. Convolutional Neural Network (CNN)
3. Classical Machine Learning Models
4. Basic scripts for bulk testing

We cover each part independently such to reproduce the results in the paper.

### Training + Validation Dataset
The dataset used for training was the Billboard Top 100 for the years 1960 to 2020. We obtained the `.wav` files for the top 10 songs in each year, which provides sufficient training data for the models. The audio files are not included in this repository or in general due to copyright restrictions, but they were obtained legally for the purposes of this project.


### Convolutional Neural Network (CNN)
The base implementation of this work was based on the <a href="">blah</a> repository, with adaptions made to accomodate for not using a classifier, but instead regression. In order to install, the following must be done:

1. Install Python3.8
2. Install NumPy, SciPy along with PyTorch

Afterward, the repository found in `convolutional_neural_network` that contains the required code.

In order to train the repository:
1. Configure `config.py` to have the proper `DATAPATH` that points to the dataset's folder
   - Structure the folders such that each folder is enclosed in a folder named after its release year
2. Invoke `python3 train.py`, which should train the model with the provided audio data from `DATAPATH`.

Note that the first time the script is run, we need to uncomment the following lines (14-18):

```python
data = Data(GENRES, DATAPATH)
data.make_raw_data()
data.save()
data = Data(GENRES, DATAPATH)
data.load()
```

3. We can then test the music release year prediction using the following command:

```sh
python3 ./genre.py <pathtofilename>
```

In the end, the expected classification for each chunk is outputted along with the final predicted year.

### Classical Machine Learning Models (Linear Regression, Decision Tree, Random Forest, Support Vector Regression)
The original implementation for music genre classification was based on the <a href="">gtzan</a> repository. Each of the models in the original repository were changed to support regression rather than classification. The logstic regression was changed to linear regression, decision tree classification was changed to decision tree regression, random forest classification was changed to random forest regression, and support vector machine was changed to support vector regression.

1. Install Python3.7
2. Install Latest SKLearn, Pandas, IPython, NBConvert

In order to train the models:
1. Go to the classical_models directory and invoke `python3 -m IPython train_classical_models.py`. This will extract the features from the songs in the music repository and create models for linear regression, decision trees, random forests, and support vector regression.

In order to run the models:
1. We can then test the music year prediction using the following command:

```sh
/usr/bin/python3 app.py -t ml -m ../models/pipe_lr.joblib -s <path_to_song>
```
The final predicted year is outputted for the specified song.
2. The `../models/pipe_lr.joblib` can be modified to use decision trees (pipe_cart), random forest (pipe_rf), or support vector regression (pipe_svr)


### Scripts

Several helper scripts are included that facilitate bulk testing. 
- `test.sh` expects the filepath for running the Python script to predict the genre
   - Note: it requires the final outputted statement to be in the format `Predicted: <ans>`
     - In addition, the location of the folders may need to be tweaked in line 24
   - The result emitted includes:
       - `results-csv.txt` which includes `predicted_year,year` pairs in a CSV
       - `sum.txt` which includes a formula denoting the absolute value differences
- `calc.py` is used to interpret the `sum.txt` and `sum_den.txt` values (more specifically computing the file percentage)
