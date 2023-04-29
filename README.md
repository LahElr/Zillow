# Zillow
My code for [Zillow’s Home Value Prediction (Zestimate)](https://www.kaggle.com/competitions/zillow-prize-1/overview) on Kaggle : 

* tried catboost and lightgbm

* did super-parameter searching

* referenced the code [here](https://www.kaggle.com/code/deepakk92/notebook211fdc91df/notebook) for parameter searching start point and basic feature processes

* although the authors of the reference recommanded to use the result of October as the result of all months as the quoted comment they wrote, I have tested that no significant result decrease is made for using precise month data.

> 11 & 12 lead to unstable results, probably due to the fact that there are few training examples for them

* added my own feature engineering

* final result:

| score(private/public) | lightgbm        | catboost        |
|:---------------------:|:---------------:|:---------------:|
| October result        | 0.07519/0.06442 | 0.0752/0.06442  |
| precise month         | 0.07628/0.06559 | 0.07622/0.06546 |

* how to run:
  * make sure to install all required libraries
  * run `python run.py`
