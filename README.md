[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Finding+charity+ML)](https://git.io/typing-svg)

<h1 align="center">Testing ML algorithms in the binary classification problem of finding people inclined to charity</h1>

<p>
    <img src="https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
    <img src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white">
</p>

<h2>Description</h2>
<p align="center"><img src="https://miro.medium.com/max/823/1*EdsKuKwKwJNQuwLDxn6n5g.png" width="1000" height="400"></p>
<p>In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. 
  Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this 
  implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, 
  where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether 
  or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can 
  (as we will see) infer this value from other publically available features.
</p>
<p>
  The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the 
  article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi online. 
  The data we investigate here consists of small changes to the original dataset, such as removing the <font color="red" size="5">fnlwgt</font> feature and records with missing or ill-formatted entries.
</p>
<h2>Exploring the data</h2>
<p>
    Note that the last column from this dataset, 'income', will be our target label (whether an individual makes more than, or at most, $50,000 annually). 
    All other columns are features about each individual in the census database.
</p>

```python
#       imports
# ================================================================
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs
# Pretty display for notebooks
%matplotlib inline

data = pd.read_csv("../input/Finding_charity_ML/ds_ml/census.csv")

# Success - Display the first record
display(data.head(n=5))
```


