# Due diligence: study and prepare the data


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# import seaborn as sns  # to plot the heat maps
import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import linalg as LA
from tqdm import trange, tqdm_notebook
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
```

## Problem background and motivation
The problem at hand consists on mapping the *signals* (i.e. information) of illiquid assets to signals of liquid assets. The idea here is of course that one prefers to hold liquid assets over illiquid assets. The reason for this is actually more profund than just because you then have a tighter control on what you can do with your assets and when. For instance, one can easily see that when you are limited by liquidity, your initial investment cannot be as large as you wish, so if you want to invest a large sum you will have to dosify it during a long period of time. But then your total investment is not benefiting from the exponential growth from the start but progressively over a long period. Mathematically, for a total investment $I_0$ on a liquid asset with return $R$ over a time $t$ one has
$$
I_t^L = I_0 R^t = \frac{I_0}{n}(R^t + R^t + \cdots).
$$
However, for an illiquid asset one is forced to split this investment into $n$ chunks and one yields instead
$$
I_t^I = \frac{I_0}{n} R^t + \frac{I_0}{n} R^{t-1} + \cdots = \frac{I_0}{n}(R^t + R^{t-1}+ \cdots),
$$
which clearly makes it a worse investment!

 Thus, it would be great if we can predict, using available signals from illiquid assets, signals of liquid assets. One simple example of this might be predicting how a (liquid) real state fund will behave based on the information available on the housing market (which is more illiquid). In this way you dont need to get into the difficult position of owning (and managing) houses and instead you invest on the liquid shares of real states funds whose value will correlate with the housing market you studied.

## Problem statement
As explained above, we are looking for a map $f:\text{iliquid}\rightarrow \text{liquid}$. So what do we exactly have? Well we have information on $M$ liquid assets, which we want to predict, and $N$ iliquid assets, which will be used to predict. In particular, we have two different type of quantities for each liquid and illiquid asset at our disposal: the daily return and a unique set of industry group identifiers which tells us roughly in which sector each asset is. Mathematically we have:
<center>

|               | Liquid | Illiquid |
| :------------ | :------: | :----: |
| Returns        |   $\{ Y_j^t \in \mathbb{R} \} \text{ for } t\in \{t_1, \cdots, t_T\} \text{ and } j \in \{1,\cdots,M\}$   |  $\{ X_j^t \in \mathbb{R}\} \text{ for }  t\in \{t_1, \cdots, t_T\} \text{ and } j \in \{1,\cdots,N\}$ |
| Industry group        |   $\{ A_n^j \in \mathbb{Z^+}\} \text{ for } j \in \{1,\cdots,M\} \text{ and } n \in \{1,\cdots,4\} $   |   $\{ B_n^j \in \mathbb{Z^+} \} \text{ for } j \in \{1,\cdots, N\}  \text{ and } n \in \{1,\cdots,4\} $ |
</center>

where note that neither $A^j_n$ nor $B^j_n$  depend on the time $t$.

There is just one small catch: we are told that

> the dates are randomized and anonymized so there is no continuity or link between any dates.

 Thus, there is not much point in making $Y$ depend on $t$ so instead, we are going to treat each different day as an independent observation of the same random process for the purposes of training. In fact, the days inside the test data are not even present in the training data. So a priori, we propose the following map
$$ 
Y_j = f(X_1,\cdots,X_N; A^j_1,\cdots,A^j_4, B_1^1,B^1_2,\cdots,B^2_1,\cdots,B_N^4 ),  \text{ for each }  j\in \{1,\cdots,M\} 
$$

where this expression is understood to be implicitly evaluated for the same $t$. 

Note that we are here assumming that knowing what $A^j_n$'s the other $Y_{k\neq j}$ have shouldnt influence on how $Y_j$ is correlated with the iliquid data. Whether or not this assumption is entirely correct might be beyond the question as, at least a priori, it might be a source of overfitting.

So where do we start? Which model do we choose? Let's first have a look at the data. 

## Loading the data & first observations
Let's have a look at what we have:


```python
# import data
X_train = pd.read_csv('X_train.csv')
X_supp = pd.read_csv('supplementary_data.csv')
Y_train = pd.read_csv('Y_train.csv')

X_test = pd.read_csv('X_test.csv')

X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>ID_DAY</th>
      <th>RET_216</th>
      <th>RET_238</th>
      <th>RET_45</th>
      <th>RET_295</th>
      <th>RET_230</th>
      <th>RET_120</th>
      <th>RET_188</th>
      <th>RET_260</th>
      <th>...</th>
      <th>RET_122</th>
      <th>RET_194</th>
      <th>RET_72</th>
      <th>RET_293</th>
      <th>RET_281</th>
      <th>RET_193</th>
      <th>RET_95</th>
      <th>RET_162</th>
      <th>RET_297</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>217</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>267095</th>
      <td>267095</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>241</td>
    </tr>
    <tr>
      <th>267096</th>
      <td>267096</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>214</td>
    </tr>
    <tr>
      <th>267097</th>
      <td>267097</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>102</td>
    </tr>
    <tr>
      <th>267098</th>
      <td>267098</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>145</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>267099</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 103 columns</p>
</div>




```python
Y_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.022351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.011892</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-0.015285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-0.019226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.006644</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>267095</th>
      <td>267095</td>
      <td>0.002080</td>
    </tr>
    <tr>
      <th>267096</th>
      <td>267096</td>
      <td>-0.002565</td>
    </tr>
    <tr>
      <th>267097</th>
      <td>267097</td>
      <td>-0.018406</td>
    </tr>
    <tr>
      <th>267098</th>
      <td>267098</td>
      <td>0.045101</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>267099</td>
      <td>0.005056</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 2 columns</p>
</div>




```python
X_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>ID_DAY</th>
      <th>RET_216</th>
      <th>RET_238</th>
      <th>RET_45</th>
      <th>RET_295</th>
      <th>RET_230</th>
      <th>RET_120</th>
      <th>RET_188</th>
      <th>RET_260</th>
      <th>...</th>
      <th>RET_122</th>
      <th>RET_194</th>
      <th>RET_72</th>
      <th>RET_293</th>
      <th>RET_281</th>
      <th>RET_193</th>
      <th>RET_95</th>
      <th>RET_162</th>
      <th>RET_297</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267100</td>
      <td>83</td>
      <td>0.043712</td>
      <td>0.020260</td>
      <td>0.027425</td>
      <td>NaN</td>
      <td>0.006963</td>
      <td>0.000528</td>
      <td>0.027680</td>
      <td>0.037824</td>
      <td>...</td>
      <td>0.016991</td>
      <td>0.022084</td>
      <td>-0.006699</td>
      <td>0.017606</td>
      <td>0.005505</td>
      <td>-0.000410</td>
      <td>0.018637</td>
      <td>0.020723</td>
      <td>0.018418</td>
      <td>139</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267101</td>
      <td>83</td>
      <td>0.043712</td>
      <td>0.020260</td>
      <td>0.027425</td>
      <td>NaN</td>
      <td>0.006963</td>
      <td>0.000528</td>
      <td>0.027680</td>
      <td>0.037824</td>
      <td>...</td>
      <td>0.016991</td>
      <td>0.022084</td>
      <td>-0.006699</td>
      <td>0.017606</td>
      <td>0.005505</td>
      <td>-0.000410</td>
      <td>0.018637</td>
      <td>0.020723</td>
      <td>0.018418</td>
      <td>129</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267102</td>
      <td>83</td>
      <td>0.043712</td>
      <td>0.020260</td>
      <td>0.027425</td>
      <td>NaN</td>
      <td>0.006963</td>
      <td>0.000528</td>
      <td>0.027680</td>
      <td>0.037824</td>
      <td>...</td>
      <td>0.016991</td>
      <td>0.022084</td>
      <td>-0.006699</td>
      <td>0.017606</td>
      <td>0.005505</td>
      <td>-0.000410</td>
      <td>0.018637</td>
      <td>0.020723</td>
      <td>0.018418</td>
      <td>136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267103</td>
      <td>83</td>
      <td>0.043712</td>
      <td>0.020260</td>
      <td>0.027425</td>
      <td>NaN</td>
      <td>0.006963</td>
      <td>0.000528</td>
      <td>0.027680</td>
      <td>0.037824</td>
      <td>...</td>
      <td>0.016991</td>
      <td>0.022084</td>
      <td>-0.006699</td>
      <td>0.017606</td>
      <td>0.005505</td>
      <td>-0.000410</td>
      <td>0.018637</td>
      <td>0.020723</td>
      <td>0.018418</td>
      <td>161</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267104</td>
      <td>83</td>
      <td>0.043712</td>
      <td>0.020260</td>
      <td>0.027425</td>
      <td>NaN</td>
      <td>0.006963</td>
      <td>0.000528</td>
      <td>0.027680</td>
      <td>0.037824</td>
      <td>...</td>
      <td>0.016991</td>
      <td>0.022084</td>
      <td>-0.006699</td>
      <td>0.017606</td>
      <td>0.005505</td>
      <td>-0.000410</td>
      <td>0.018637</td>
      <td>0.020723</td>
      <td>0.018418</td>
      <td>217</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114463</th>
      <td>381563</td>
      <td>415</td>
      <td>-0.018322</td>
      <td>0.034389</td>
      <td>0.044101</td>
      <td>-0.033431</td>
      <td>-0.026487</td>
      <td>-0.012058</td>
      <td>0.041611</td>
      <td>0.003381</td>
      <td>...</td>
      <td>-0.005801</td>
      <td>-0.011807</td>
      <td>-0.022010</td>
      <td>-0.010799</td>
      <td>-0.023107</td>
      <td>-0.002279</td>
      <td>-0.015246</td>
      <td>-0.016463</td>
      <td>-0.000903</td>
      <td>241</td>
    </tr>
    <tr>
      <th>114464</th>
      <td>381564</td>
      <td>415</td>
      <td>-0.018322</td>
      <td>0.034389</td>
      <td>0.044101</td>
      <td>-0.033431</td>
      <td>-0.026487</td>
      <td>-0.012058</td>
      <td>0.041611</td>
      <td>0.003381</td>
      <td>...</td>
      <td>-0.005801</td>
      <td>-0.011807</td>
      <td>-0.022010</td>
      <td>-0.010799</td>
      <td>-0.023107</td>
      <td>-0.002279</td>
      <td>-0.015246</td>
      <td>-0.016463</td>
      <td>-0.000903</td>
      <td>214</td>
    </tr>
    <tr>
      <th>114465</th>
      <td>381565</td>
      <td>415</td>
      <td>-0.018322</td>
      <td>0.034389</td>
      <td>0.044101</td>
      <td>-0.033431</td>
      <td>-0.026487</td>
      <td>-0.012058</td>
      <td>0.041611</td>
      <td>0.003381</td>
      <td>...</td>
      <td>-0.005801</td>
      <td>-0.011807</td>
      <td>-0.022010</td>
      <td>-0.010799</td>
      <td>-0.023107</td>
      <td>-0.002279</td>
      <td>-0.015246</td>
      <td>-0.016463</td>
      <td>-0.000903</td>
      <td>102</td>
    </tr>
    <tr>
      <th>114466</th>
      <td>381566</td>
      <td>415</td>
      <td>-0.018322</td>
      <td>0.034389</td>
      <td>0.044101</td>
      <td>-0.033431</td>
      <td>-0.026487</td>
      <td>-0.012058</td>
      <td>0.041611</td>
      <td>0.003381</td>
      <td>...</td>
      <td>-0.005801</td>
      <td>-0.011807</td>
      <td>-0.022010</td>
      <td>-0.010799</td>
      <td>-0.023107</td>
      <td>-0.002279</td>
      <td>-0.015246</td>
      <td>-0.016463</td>
      <td>-0.000903</td>
      <td>145</td>
    </tr>
    <tr>
      <th>114467</th>
      <td>381567</td>
      <td>415</td>
      <td>-0.018322</td>
      <td>0.034389</td>
      <td>0.044101</td>
      <td>-0.033431</td>
      <td>-0.026487</td>
      <td>-0.012058</td>
      <td>0.041611</td>
      <td>0.003381</td>
      <td>...</td>
      <td>-0.005801</td>
      <td>-0.011807</td>
      <td>-0.022010</td>
      <td>-0.010799</td>
      <td>-0.023107</td>
      <td>-0.002279</td>
      <td>-0.015246</td>
      <td>-0.016463</td>
      <td>-0.000903</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
<p>114468 rows × 103 columns</p>
</div>




```python
X_supp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_asset</th>
      <th>CLASS_LEVEL_1</th>
      <th>CLASS_LEVEL_2</th>
      <th>CLASS_LEVEL_3</th>
      <th>CLASS_LEVEL_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>216</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>238</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>3</td>
      <td>5</td>
      <td>20</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>295</td>
      <td>10</td>
      <td>22</td>
      <td>49</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>230</td>
      <td>4</td>
      <td>10</td>
      <td>28</td>
      <td>47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>241</td>
      <td>3</td>
      <td>8</td>
      <td>26</td>
      <td>42</td>
    </tr>
    <tr>
      <th>196</th>
      <td>214</td>
      <td>2</td>
      <td>2</td>
      <td>13</td>
      <td>22</td>
    </tr>
    <tr>
      <th>197</th>
      <td>102</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>12</td>
    </tr>
    <tr>
      <th>198</th>
      <td>145</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>20</td>
    </tr>
    <tr>
      <th>199</th>
      <td>155</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 5 columns</p>
</div>



Note this data, as is, looks like it can definetely be a source of problems if we dont handle the different scales among the different `CLASS_LEVEL_n`s. Indeed observe that they have wildly different distributions.


```python
X_supp.drop(columns=["ID_asset"]).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CLASS_LEVEL_1</th>
      <th>CLASS_LEVEL_2</th>
      <th>CLASS_LEVEL_3</th>
      <th>CLASS_LEVEL_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.300000</td>
      <td>9.280000</td>
      <td>24.680000</td>
      <td>39.620000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.162278</td>
      <td>7.858139</td>
      <td>16.276431</td>
      <td>24.468123</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>24.500000</td>
      <td>38.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>17.000000</td>
      <td>40.000000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>22.000000</td>
      <td>49.000000</td>
      <td>77.000000</td>
    </tr>
  </tbody>
</table>
</div>



This is important because, as we teased, we will want to weight the iliquid returns `RET_n`s with their own `CLASS_LEVEL_` and the target's. We will describe this feature engenieering in more detail later on, but for now take my word that we need to standarise the industry classes:


```python
X_supp[['CLASS_LEVEL_1', 'CLASS_LEVEL_2',"CLASS_LEVEL_3","CLASS_LEVEL_4"]] = StandardScaler().fit_transform(X_supp[['CLASS_LEVEL_1', 'CLASS_LEVEL_2',"CLASS_LEVEL_3","CLASS_LEVEL_4"]])
rows_norm = np.reshape( LA.norm(X_supp.drop(columns=["ID_asset"]).to_numpy(),axis=1),(200,1))

X_supp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_asset</th>
      <th>CLASS_LEVEL_1</th>
      <th>CLASS_LEVEL_2</th>
      <th>CLASS_LEVEL_3</th>
      <th>CLASS_LEVEL_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>216</td>
      <td>-0.729149</td>
      <td>-0.928753</td>
      <td>-0.780995</td>
      <td>-0.803872</td>
    </tr>
    <tr>
      <th>1</th>
      <td>238</td>
      <td>-0.729149</td>
      <td>-0.928753</td>
      <td>-0.780995</td>
      <td>-0.762900</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>-0.412128</td>
      <td>-0.546025</td>
      <td>-0.288254</td>
      <td>-0.312207</td>
    </tr>
    <tr>
      <th>3</th>
      <td>295</td>
      <td>1.807021</td>
      <td>1.622766</td>
      <td>1.497935</td>
      <td>1.531536</td>
    </tr>
    <tr>
      <th>4</th>
      <td>230</td>
      <td>-0.095106</td>
      <td>0.091855</td>
      <td>0.204488</td>
      <td>0.302374</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>241</td>
      <td>-0.412128</td>
      <td>-0.163297</td>
      <td>0.081302</td>
      <td>0.097514</td>
    </tr>
    <tr>
      <th>196</th>
      <td>214</td>
      <td>-0.729149</td>
      <td>-0.928753</td>
      <td>-0.719403</td>
      <td>-0.721928</td>
    </tr>
    <tr>
      <th>197</th>
      <td>102</td>
      <td>-1.046170</td>
      <td>-1.056329</td>
      <td>-1.212144</td>
      <td>-1.131648</td>
    </tr>
    <tr>
      <th>198</th>
      <td>145</td>
      <td>-0.729149</td>
      <td>-0.928753</td>
      <td>-0.780995</td>
      <td>-0.803872</td>
    </tr>
    <tr>
      <th>199</th>
      <td>155</td>
      <td>-0.729149</td>
      <td>-0.928753</td>
      <td>-1.088959</td>
      <td>-1.049704</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 5 columns</p>
</div>



Let's now look at `X_train` and `Y_train` again. It is clear that the data needs some resorting: note that most rows are repeated since the `RET_n` columns have the same value for the same day. This is because the data has been structured so that each row connects to one particular `ID_TARGET` but this doesnt add any information other than knowing whether or not there is data missing so let's reorganise this in the next section. 

First observe we have data fo $2748$ days which is, a prior, too little data for certain ML algorithms like deep neural networks. However, we will see later on that we can remedy this.


```python
# number of days for which we have data
n_days = len(X_train["ID_DAY"].unique())
n_days
```




    2748



Furthermore, as we discussed in the problem statement section, note that in the `X_test` data we have values of `ID_DAY` with data not in `X_train`:


```python
# ID_DAY = 415  is in X_test but not in X_train
415 in X_train["ID_DAY"].unique()
```




    False



So we need to be careful with how we treat `ID_DAY`. So far, aside of to separate the different observations of all the `RET_n` it should not play any significant predictive role. 

# Cleaning and reorganising the data
### Model considerations
Armed with the lessons learned above, let's begin by reorganising the data. How do we want the data to be organised? Obviously we want the train data to contain all necessary information but how that information is arranged depends a bit on what model we will use. 

Given the problem and the scarcity of data, it is hard not to start with a linear model of some sort. Thus we could propose 
$$
\hat{Y}_j = \sum_k X_k \beta^k_j + \sum_{lk} B_k^l \alpha_{lj}^k + \sum_k A_k^j \gamma_j^k + m_j, 
$$
where we again implicitly assume $Y_j$ and $X_k$ are at the same time $t$ and matrices $B_k^l$ and $A_k^j$ store all the asset coefficients. One could reasonably argue that in fact only the difference between $B$ and $A$ are important as this difference tells you how relevant one particular $X_k$ given how similar is for a given $Y_j$. We propose that models like 
$$
\hat{Y}_j = \sum_{kn} \frac{X_k}{1 + (B_n^k-A_n^j)^2} \beta^{kn}_j + m_j, 
$$
are very reasonable and deserve full attention. Think about it, if $B_n^k\approx A_n^j$ then $X_k$ is left untouch, however if $B_n^k\neq A_n^j$ then $X_k$ is suppressed by their difference squared (the $1$ just avoids $1/0$).  While the above expression is mathematically sound, it is a bit cumbersome to deal with since we are dealing with higher rank tensors for no good reason. We are essentially doing feature engineering so we can instead redefine what the features are by defining a new vector of features for each target $Y_j$
$$
\vec{F_j} = \left(1,\frac{X_1}{1+(B_1^1-A_1^j)^2},\frac{X_1}{1+(B_2^1-A_2^j)^2},\cdots, \frac{X_2}{1+(B_1^2-A_1^j)^2}, \cdots,\frac{X_N}{1+(B_4^N-A_4^j)^2} \right), \quad \text{where} \quad  \vec{F_j} \in \mathbb{R}^P
$$
so that
$$
\hat{Y}_j = \vec{F_j} \cdot \vec{\beta_j}, \quad \text{where} \quad \vec{\beta_j}[0]=m_j.
$$
Now this model is manifestly of the form of the linear models we are used to deal with. From now on, I will drop index $j$ as it is clear that with this model nor the loss nor the predicted $Y_j$ depend on anything with $k\neq j$.  Now if we use a mean squared error as our loss function then we already know that
$$
\mathcal{L} = \frac{1}{2S} \sum_\alpha^S (Y^\alpha - \hat{Y}^\alpha)^2 \equiv \frac{1}{2S} \sum_\alpha^S (Y^\alpha -  \vec{F}^\alpha \cdot \vec{\beta})^2 \quad \text{ then } \quad \partial_{\beta_j}\mathcal{L} = 0 \implies \vec{\beta} = (F^T F)^{-1} F^T \vec{y},
$$
where again here $\vec{\beta}$ is the entries of what we denoted above $\vec{\beta_j}$ for some particular $j$ and $F\in \mathbb{R}^{S\times P}$ is the matrix with $S$ rows (one for each observation) an $P$ columns (one for each flattened feature, c.f. definition of  $\vec{F_j}$ above).

Before moving on I feel like there is one last technical detail worth commenting. We weighted the returns above so that assets that are different are suppressed. But, what makes two different assets far away? Well, at first glance we assumed that two assets are different if their $A$'s and $B$'s are different or in other words we essentially assumed
$$
\textrm{dist}(R_j,R_k) = \sum_n (A_n^j-B^k_n)^2,
$$
but maybe there are some $A_n^j$'s that contribute more in distinguishing assets, so shouldnt we find out a set of $\beta_m$'s to weight the above euclidian distance
$$
\textrm{dist}(R_j,R_k) = \sum_n \beta_n (A_n^j-B^k_n)^2,
$$
what about other distance functions? Well first of all, the answer is we have implicitly already fitted those $\beta_m$'s. The reason behind why we have split each return into $4$ different weighted returns is to allow each to have its own coefficient (from the linear regression for instance). In this way, the learning algorithm will have these independent pieces of information to play with. Having said that, there are indeed alternative ways we could think of to estimate these $\beta_m$'s, for instance by cooking some loss function that demands two assets that behave similarly over all available observations to be close in some precise sense. However, we will not persue this direction now and rather focus on the bigger picture.

### Reshaping the data
As argued, it is clear that the input data to predict the $j$ -th asset should  contain all the iliquid returns `RET_n` weighted with all the iliquid class coefficients `B_n_k` and the $j$ th liquid asset class coefficients `A_n_j`. So we will stack it like
$$
X_{train} = 
\begin{bmatrix}
    F[1]_j^{t=1} &  F[2]_j^{t=1} & \cdots  &  F[P]_j^{t=1} \\
     F[1]_j^{t=2} &  F[2]_j^{t=2} & \cdots  &  F[P]_j^{t=2} \\
    \vdots & \vdots & \vdots & \vdots \\
    F[1]_j^{t=S} &  F[2]_j^{t=S} & \cdots  &  F[P]_j^{t=S} 
\end{bmatrix} \quad  \text{ and } \quad
Y_{train} = \begin{bmatrix}
    Y_j^{t=1} \\
    Y_j^{t=2} \\
    \vdots \\
    Y_j^{t=S} \\
\end{bmatrix}.
$$
where each row is an individual (day) observation. 

By looking at the data, we see we need to train $100$ individual linear models, one for each liquid asset as explained above. Let's extract the train data for one particular liquid asset from the total data and take it from there.

We begin by dropping the data on the rest of the liquid assets. 


```python
# we picked asset 139 as an example
X_train_139 = X_train.loc[X_train['ID_TARGET'] == 139]
X_train_139
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>ID_DAY</th>
      <th>RET_216</th>
      <th>RET_238</th>
      <th>RET_45</th>
      <th>RET_295</th>
      <th>RET_230</th>
      <th>RET_120</th>
      <th>RET_188</th>
      <th>RET_260</th>
      <th>...</th>
      <th>RET_122</th>
      <th>RET_194</th>
      <th>RET_72</th>
      <th>RET_293</th>
      <th>RET_281</th>
      <th>RET_193</th>
      <th>RET_95</th>
      <th>RET_162</th>
      <th>RET_297</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>NaN</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>139</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>3355</td>
      <td>0.025848</td>
      <td>-0.002109</td>
      <td>-0.021802</td>
      <td>0.040229</td>
      <td>0.015093</td>
      <td>-0.015498</td>
      <td>0.011188</td>
      <td>0.011622</td>
      <td>...</td>
      <td>0.001506</td>
      <td>0.003077</td>
      <td>-0.002341</td>
      <td>0.061228</td>
      <td>0.005301</td>
      <td>0.008942</td>
      <td>-0.010232</td>
      <td>0.005529</td>
      <td>0.006545</td>
      <td>139</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>1662</td>
      <td>-0.012267</td>
      <td>0.007461</td>
      <td>0.051311</td>
      <td>0.105340</td>
      <td>-0.006361</td>
      <td>0.004964</td>
      <td>0.011933</td>
      <td>0.018921</td>
      <td>...</td>
      <td>0.009226</td>
      <td>0.032670</td>
      <td>-0.000716</td>
      <td>0.008429</td>
      <td>-0.002871</td>
      <td>0.009932</td>
      <td>0.023721</td>
      <td>0.009349</td>
      <td>0.033554</td>
      <td>139</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>3405</td>
      <td>-0.033598</td>
      <td>-0.003446</td>
      <td>-0.009100</td>
      <td>0.016753</td>
      <td>-0.016952</td>
      <td>-0.008924</td>
      <td>-0.010984</td>
      <td>0.001948</td>
      <td>...</td>
      <td>0.001551</td>
      <td>-0.008077</td>
      <td>0.009507</td>
      <td>-0.008005</td>
      <td>-0.016593</td>
      <td>-0.007481</td>
      <td>-0.024835</td>
      <td>0.004360</td>
      <td>0.031087</td>
      <td>139</td>
    </tr>
    <tr>
      <th>400</th>
      <td>400</td>
      <td>1602</td>
      <td>0.029170</td>
      <td>-0.084293</td>
      <td>0.000325</td>
      <td>0.021458</td>
      <td>-0.018345</td>
      <td>-0.012230</td>
      <td>-0.001818</td>
      <td>0.000842</td>
      <td>...</td>
      <td>-0.012181</td>
      <td>0.002507</td>
      <td>0.004408</td>
      <td>-0.022913</td>
      <td>-0.010133</td>
      <td>-0.006468</td>
      <td>-0.025205</td>
      <td>-0.001612</td>
      <td>-0.056735</td>
      <td>139</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266605</th>
      <td>266605</td>
      <td>3332</td>
      <td>-0.012034</td>
      <td>-0.021459</td>
      <td>-0.004453</td>
      <td>-0.011604</td>
      <td>0.086227</td>
      <td>-0.003680</td>
      <td>-0.022954</td>
      <td>-0.011491</td>
      <td>...</td>
      <td>-0.005921</td>
      <td>-0.000173</td>
      <td>0.016216</td>
      <td>-0.010907</td>
      <td>-0.043811</td>
      <td>-0.003677</td>
      <td>0.003975</td>
      <td>-0.004490</td>
      <td>-0.005928</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266704</th>
      <td>266704</td>
      <td>2314</td>
      <td>0.001643</td>
      <td>0.018795</td>
      <td>0.052567</td>
      <td>0.003184</td>
      <td>0.102134</td>
      <td>-0.001182</td>
      <td>0.030033</td>
      <td>0.010007</td>
      <td>...</td>
      <td>0.002858</td>
      <td>0.006811</td>
      <td>0.026210</td>
      <td>-0.001437</td>
      <td>0.011124</td>
      <td>-0.000478</td>
      <td>-0.013254</td>
      <td>0.043495</td>
      <td>0.002780</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266803</th>
      <td>266803</td>
      <td>1863</td>
      <td>0.010200</td>
      <td>-0.003231</td>
      <td>-0.018298</td>
      <td>-0.009153</td>
      <td>-0.055494</td>
      <td>-0.001865</td>
      <td>-0.035667</td>
      <td>-0.020259</td>
      <td>...</td>
      <td>-0.003975</td>
      <td>-0.006258</td>
      <td>0.001214</td>
      <td>-0.003871</td>
      <td>-0.010329</td>
      <td>-0.001843</td>
      <td>0.009359</td>
      <td>-0.017191</td>
      <td>-0.012770</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266902</th>
      <td>266902</td>
      <td>2868</td>
      <td>0.000532</td>
      <td>-0.006588</td>
      <td>-0.053379</td>
      <td>-0.011904</td>
      <td>-0.068993</td>
      <td>-0.018487</td>
      <td>0.003643</td>
      <td>-0.012991</td>
      <td>...</td>
      <td>0.005227</td>
      <td>0.002653</td>
      <td>-0.045902</td>
      <td>-0.008034</td>
      <td>-0.016351</td>
      <td>-0.001056</td>
      <td>-0.018058</td>
      <td>-0.012541</td>
      <td>-0.001299</td>
      <td>139</td>
    </tr>
    <tr>
      <th>267001</th>
      <td>267001</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
<p>2739 rows × 103 columns</p>
</div>



Filling the `NaN`'s with their columns average (which is the most resonable fix I can think of at this stage)


```python
X_train_139 = X_train_139.fillna(X_train_139.mean())
X_train_139
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>ID_DAY</th>
      <th>RET_216</th>
      <th>RET_238</th>
      <th>RET_45</th>
      <th>RET_295</th>
      <th>RET_230</th>
      <th>RET_120</th>
      <th>RET_188</th>
      <th>RET_260</th>
      <th>...</th>
      <th>RET_122</th>
      <th>RET_194</th>
      <th>RET_72</th>
      <th>RET_293</th>
      <th>RET_281</th>
      <th>RET_193</th>
      <th>RET_95</th>
      <th>RET_162</th>
      <th>RET_297</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3316</td>
      <td>0.004024</td>
      <td>0.009237</td>
      <td>0.004967</td>
      <td>0.000729</td>
      <td>0.017040</td>
      <td>0.013885</td>
      <td>0.041885</td>
      <td>0.015207</td>
      <td>...</td>
      <td>0.007596</td>
      <td>0.015010</td>
      <td>0.014733</td>
      <td>-0.000476</td>
      <td>0.006539</td>
      <td>-0.010233</td>
      <td>0.001251</td>
      <td>-0.003102</td>
      <td>-0.094847</td>
      <td>139</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>3355</td>
      <td>0.025848</td>
      <td>-0.002109</td>
      <td>-0.021802</td>
      <td>0.040229</td>
      <td>0.015093</td>
      <td>-0.015498</td>
      <td>0.011188</td>
      <td>0.011622</td>
      <td>...</td>
      <td>0.001506</td>
      <td>0.003077</td>
      <td>-0.002341</td>
      <td>0.061228</td>
      <td>0.005301</td>
      <td>0.008942</td>
      <td>-0.010232</td>
      <td>0.005529</td>
      <td>0.006545</td>
      <td>139</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>1662</td>
      <td>-0.012267</td>
      <td>0.007461</td>
      <td>0.051311</td>
      <td>0.105340</td>
      <td>-0.006361</td>
      <td>0.004964</td>
      <td>0.011933</td>
      <td>0.018921</td>
      <td>...</td>
      <td>0.009226</td>
      <td>0.032670</td>
      <td>-0.000716</td>
      <td>0.008429</td>
      <td>-0.002871</td>
      <td>0.009932</td>
      <td>0.023721</td>
      <td>0.009349</td>
      <td>0.033554</td>
      <td>139</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>3405</td>
      <td>-0.033598</td>
      <td>-0.003446</td>
      <td>-0.009100</td>
      <td>0.016753</td>
      <td>-0.016952</td>
      <td>-0.008924</td>
      <td>-0.010984</td>
      <td>0.001948</td>
      <td>...</td>
      <td>0.001551</td>
      <td>-0.008077</td>
      <td>0.009507</td>
      <td>-0.008005</td>
      <td>-0.016593</td>
      <td>-0.007481</td>
      <td>-0.024835</td>
      <td>0.004360</td>
      <td>0.031087</td>
      <td>139</td>
    </tr>
    <tr>
      <th>400</th>
      <td>400</td>
      <td>1602</td>
      <td>0.029170</td>
      <td>-0.084293</td>
      <td>0.000325</td>
      <td>0.021458</td>
      <td>-0.018345</td>
      <td>-0.012230</td>
      <td>-0.001818</td>
      <td>0.000842</td>
      <td>...</td>
      <td>-0.012181</td>
      <td>0.002507</td>
      <td>0.004408</td>
      <td>-0.022913</td>
      <td>-0.010133</td>
      <td>-0.006468</td>
      <td>-0.025205</td>
      <td>-0.001612</td>
      <td>-0.056735</td>
      <td>139</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266605</th>
      <td>266605</td>
      <td>3332</td>
      <td>-0.012034</td>
      <td>-0.021459</td>
      <td>-0.004453</td>
      <td>-0.011604</td>
      <td>0.086227</td>
      <td>-0.003680</td>
      <td>-0.022954</td>
      <td>-0.011491</td>
      <td>...</td>
      <td>-0.005921</td>
      <td>-0.000173</td>
      <td>0.016216</td>
      <td>-0.010907</td>
      <td>-0.043811</td>
      <td>-0.003677</td>
      <td>0.003975</td>
      <td>-0.004490</td>
      <td>-0.005928</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266704</th>
      <td>266704</td>
      <td>2314</td>
      <td>0.001643</td>
      <td>0.018795</td>
      <td>0.052567</td>
      <td>0.003184</td>
      <td>0.102134</td>
      <td>-0.001182</td>
      <td>0.030033</td>
      <td>0.010007</td>
      <td>...</td>
      <td>0.002858</td>
      <td>0.006811</td>
      <td>0.026210</td>
      <td>-0.001437</td>
      <td>0.011124</td>
      <td>-0.000478</td>
      <td>-0.013254</td>
      <td>0.043495</td>
      <td>0.002780</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266803</th>
      <td>266803</td>
      <td>1863</td>
      <td>0.010200</td>
      <td>-0.003231</td>
      <td>-0.018298</td>
      <td>-0.009153</td>
      <td>-0.055494</td>
      <td>-0.001865</td>
      <td>-0.035667</td>
      <td>-0.020259</td>
      <td>...</td>
      <td>-0.003975</td>
      <td>-0.006258</td>
      <td>0.001214</td>
      <td>-0.003871</td>
      <td>-0.010329</td>
      <td>-0.001843</td>
      <td>0.009359</td>
      <td>-0.017191</td>
      <td>-0.012770</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266902</th>
      <td>266902</td>
      <td>2868</td>
      <td>0.000532</td>
      <td>-0.006588</td>
      <td>-0.053379</td>
      <td>-0.011904</td>
      <td>-0.068993</td>
      <td>-0.018487</td>
      <td>0.003643</td>
      <td>-0.012991</td>
      <td>...</td>
      <td>0.005227</td>
      <td>0.002653</td>
      <td>-0.045902</td>
      <td>-0.008034</td>
      <td>-0.016351</td>
      <td>-0.001056</td>
      <td>-0.018058</td>
      <td>-0.012541</td>
      <td>-0.001299</td>
      <td>139</td>
    </tr>
    <tr>
      <th>267001</th>
      <td>267001</td>
      <td>3028</td>
      <td>0.025293</td>
      <td>-0.003277</td>
      <td>-0.028823</td>
      <td>-0.006021</td>
      <td>-0.008381</td>
      <td>0.006805</td>
      <td>0.018665</td>
      <td>-0.010479</td>
      <td>...</td>
      <td>0.013698</td>
      <td>-0.007358</td>
      <td>0.022241</td>
      <td>0.008688</td>
      <td>0.006896</td>
      <td>0.005854</td>
      <td>-0.003103</td>
      <td>-0.022325</td>
      <td>0.014949</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
<p>2739 rows × 103 columns</p>
</div>



And now we need to weight the iliquid returns with the industry class information as discussed. For that we will use the already loaded `X_supp` data and define the following helper function which does that plus all the above steps:


```python
def weight_returns(x_df : pd.DataFrame, supp_df : pd.DataFrame, liquid_ID : int)->pd.DataFrame:
    """
    Transforms the illiquid returns in `x_df` dataframe by weighting them with the class information in `supp_df` of all illiquid assets and liquid asset with ID `liquid_ID`. Note this means this only applies for the data points associated with the liquid asset with ID liquid_ID. This transformation is given by
    ```
    RET_n(1+(B^n_1-A^j_1)**2),   RET_n(1+(B^n_2-A^j_2)**2), ...
    ```
    Note how this enlarges the number of features by a factor of 4.

    ### Parameters
    - x_df : pd.DataFrame
    - supp_df : pd.DataFrame
    - liquid_ID : int

    ### Returns
    - weighted_df : pd.DataFrame
    """
    # filter for liquid_ID
    train_df = x_df.loc[x_df['ID_TARGET'] == liquid_ID] 
    # handle NaNs
    train_df = train_df.fillna(train_df.mean())
    
    # get all return names
    cols_names = train_df.columns
    ret_names = []
    for name in cols_names:
        if "RET_" in name:
            ret_names.append(name)

    # 4D numpy vector of liquid classes
    liquid_classes = supp_df.loc[supp_df['ID_asset'] == liquid_ID].drop(columns=["ID_asset"]).to_numpy()[0]

    # weight the data
    temp_df = pd.DataFrame({"ID":train_df["ID"]})
    for ret_name in ret_names: # plus all the other data
        for j in range(4):
            col_name =  ret_name + "_" + str(j)
            iliquid_classes = supp_df.loc[supp_df['ID_asset'] == int(ret_name[4:])].drop(columns=["ID_asset"]).to_numpy()[0]
            
            temp_df[col_name] =  train_df[ret_name].to_numpy() / (1 + (iliquid_classes[j] - liquid_classes[j])**2)
    return temp_df

```

Let's test it with liquid target id `139`:


```python
weight_returns(X_train,X_supp,139)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_216_0</th>
      <th>RET_216_1</th>
      <th>RET_216_2</th>
      <th>RET_216_3</th>
      <th>RET_238_0</th>
      <th>RET_238_1</th>
      <th>RET_238_2</th>
      <th>RET_238_3</th>
      <th>RET_45_0</th>
      <th>...</th>
      <th>RET_95_2</th>
      <th>RET_95_3</th>
      <th>RET_162_0</th>
      <th>RET_162_1</th>
      <th>RET_162_2</th>
      <th>RET_162_3</th>
      <th>RET_297_0</th>
      <th>RET_297_1</th>
      <th>RET_297_2</th>
      <th>RET_297_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.002870</td>
      <td>0.003778</td>
      <td>0.002758</td>
      <td>0.002506</td>
      <td>0.006588</td>
      <td>0.008672</td>
      <td>0.006331</td>
      <td>0.005526</td>
      <td>0.002608</td>
      <td>...</td>
      <td>0.000133</td>
      <td>0.000129</td>
      <td>-0.001189</td>
      <td>-0.001181</td>
      <td>-0.000824</td>
      <td>-0.000705</td>
      <td>-0.020538</td>
      <td>-0.022636</td>
      <td>-0.018484</td>
      <td>-0.016595</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>0.018436</td>
      <td>0.024268</td>
      <td>0.017716</td>
      <td>0.016094</td>
      <td>-0.001504</td>
      <td>-0.001980</td>
      <td>-0.001446</td>
      <td>-0.001262</td>
      <td>-0.011448</td>
      <td>...</td>
      <td>-0.001091</td>
      <td>-0.001055</td>
      <td>0.002120</td>
      <td>0.002104</td>
      <td>0.001468</td>
      <td>0.001257</td>
      <td>0.001417</td>
      <td>0.001562</td>
      <td>0.001275</td>
      <td>0.001145</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>-0.008749</td>
      <td>-0.011517</td>
      <td>-0.008407</td>
      <td>-0.007638</td>
      <td>0.005322</td>
      <td>0.007005</td>
      <td>0.005114</td>
      <td>0.004464</td>
      <td>0.026942</td>
      <td>...</td>
      <td>0.002529</td>
      <td>0.002445</td>
      <td>0.003585</td>
      <td>0.003558</td>
      <td>0.002483</td>
      <td>0.002125</td>
      <td>0.007266</td>
      <td>0.008008</td>
      <td>0.006539</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>-0.023964</td>
      <td>-0.031545</td>
      <td>-0.023028</td>
      <td>-0.020920</td>
      <td>-0.002458</td>
      <td>-0.003235</td>
      <td>-0.002362</td>
      <td>-0.002062</td>
      <td>-0.004778</td>
      <td>...</td>
      <td>-0.002648</td>
      <td>-0.002560</td>
      <td>0.001672</td>
      <td>0.001660</td>
      <td>0.001158</td>
      <td>0.000991</td>
      <td>0.006732</td>
      <td>0.007419</td>
      <td>0.006058</td>
      <td>0.005439</td>
    </tr>
    <tr>
      <th>400</th>
      <td>400</td>
      <td>0.020806</td>
      <td>0.027387</td>
      <td>0.019992</td>
      <td>0.018163</td>
      <td>-0.060123</td>
      <td>-0.079141</td>
      <td>-0.057773</td>
      <td>-0.050430</td>
      <td>0.000171</td>
      <td>...</td>
      <td>-0.002687</td>
      <td>-0.002598</td>
      <td>-0.000618</td>
      <td>-0.000614</td>
      <td>-0.000428</td>
      <td>-0.000366</td>
      <td>-0.012285</td>
      <td>-0.013541</td>
      <td>-0.011057</td>
      <td>-0.009927</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266605</th>
      <td>266605</td>
      <td>-0.008584</td>
      <td>-0.011299</td>
      <td>-0.008248</td>
      <td>-0.007493</td>
      <td>-0.015306</td>
      <td>-0.020148</td>
      <td>-0.014708</td>
      <td>-0.012839</td>
      <td>-0.002338</td>
      <td>...</td>
      <td>0.000424</td>
      <td>0.000410</td>
      <td>-0.001722</td>
      <td>-0.001709</td>
      <td>-0.001192</td>
      <td>-0.001021</td>
      <td>-0.001284</td>
      <td>-0.001415</td>
      <td>-0.001155</td>
      <td>-0.001037</td>
    </tr>
    <tr>
      <th>266704</th>
      <td>266704</td>
      <td>0.001172</td>
      <td>0.001543</td>
      <td>0.001126</td>
      <td>0.001023</td>
      <td>0.013406</td>
      <td>0.017647</td>
      <td>0.012882</td>
      <td>0.011245</td>
      <td>0.027601</td>
      <td>...</td>
      <td>-0.001413</td>
      <td>-0.001366</td>
      <td>0.016677</td>
      <td>0.016553</td>
      <td>0.011551</td>
      <td>0.009887</td>
      <td>0.000602</td>
      <td>0.000663</td>
      <td>0.000542</td>
      <td>0.000486</td>
    </tr>
    <tr>
      <th>266803</th>
      <td>266803</td>
      <td>0.007275</td>
      <td>0.009577</td>
      <td>0.006991</td>
      <td>0.006351</td>
      <td>-0.002305</td>
      <td>-0.003034</td>
      <td>-0.002215</td>
      <td>-0.001933</td>
      <td>-0.009608</td>
      <td>...</td>
      <td>0.000998</td>
      <td>0.000965</td>
      <td>-0.006591</td>
      <td>-0.006543</td>
      <td>-0.004565</td>
      <td>-0.003908</td>
      <td>-0.002765</td>
      <td>-0.003048</td>
      <td>-0.002489</td>
      <td>-0.002234</td>
    </tr>
    <tr>
      <th>266902</th>
      <td>266902</td>
      <td>0.000379</td>
      <td>0.000499</td>
      <td>0.000364</td>
      <td>0.000331</td>
      <td>-0.004699</td>
      <td>-0.006185</td>
      <td>-0.004515</td>
      <td>-0.003941</td>
      <td>-0.028027</td>
      <td>...</td>
      <td>-0.001925</td>
      <td>-0.001861</td>
      <td>-0.004809</td>
      <td>-0.004773</td>
      <td>-0.003330</td>
      <td>-0.002851</td>
      <td>-0.000281</td>
      <td>-0.000310</td>
      <td>-0.000253</td>
      <td>-0.000227</td>
    </tr>
    <tr>
      <th>267001</th>
      <td>267001</td>
      <td>0.018040</td>
      <td>0.023747</td>
      <td>0.017335</td>
      <td>0.015749</td>
      <td>-0.002337</td>
      <td>-0.003076</td>
      <td>-0.002246</td>
      <td>-0.001960</td>
      <td>-0.015134</td>
      <td>...</td>
      <td>-0.000331</td>
      <td>-0.000320</td>
      <td>-0.008560</td>
      <td>-0.008496</td>
      <td>-0.005929</td>
      <td>-0.005074</td>
      <td>0.003237</td>
      <td>0.003568</td>
      <td>0.002913</td>
      <td>0.002616</td>
    </tr>
  </tbody>
</table>
<p>2739 rows × 401 columns</p>
</div>



Sanity check: the first entry should be `RET_216/(1+(B^216_1-A^139_1)**2)` which is indeed $0.004024/(1+(-0.729149-(-1.363192))^2)\approx 0.00287016$. Now for the training `Y_train` we simply do 


```python
# get relevant IDs
ID_list = weight_returns(X_train,X_supp,139)["ID"]
# select those relevant IDs 
Y_train.loc[Y_train['ID'].isin(ID_list)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.022351</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>0.008354</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>0.012218</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>-0.004456</td>
    </tr>
    <tr>
      <th>400</th>
      <td>400</td>
      <td>0.008788</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266605</th>
      <td>266605</td>
      <td>0.019691</td>
    </tr>
    <tr>
      <th>266704</th>
      <td>266704</td>
      <td>0.001614</td>
    </tr>
    <tr>
      <th>266803</th>
      <td>266803</td>
      <td>0.003476</td>
    </tr>
    <tr>
      <th>266902</th>
      <td>266902</td>
      <td>-0.002670</td>
    </tr>
    <tr>
      <th>267001</th>
      <td>267001</td>
      <td>-0.016508</td>
    </tr>
  </tbody>
</table>
<p>2739 rows × 2 columns</p>
</div>



which is clearly doing what we want. So we can convine these two methods in one single handy function:


```python
def data_to_train(x_df : pd.DataFrame, y_df : pd.DataFrame, supp_df : pd.DataFrame,liquid_ID : int) -> tuple:
    """
    Transforms the illiquid returns in `x_df` using `weight_returns(x_df,supp_df,liquid_ID)` and reformats `y_df` to match this appropiately.

    ### Parameters
    - x_df : pd.DataFrame
    - y_df : pd.DataFrame
    - supp_df : pd.DataFrame
    - liquid_ID : int

    ### Returns
    - (x_array, y_array) : tuple of numpy arrays
    """
    # get weighted data using `weight_returns` function
    train_x_df = weight_returns(x_df,supp_df,liquid_ID)

    # select relevant liquid returns
    ID_list = train_x_df["ID"]
    train_y_df = y_df.loc[y_df['ID'].isin(ID_list)]    

    # drop IDs (useless for training now) and convert to np array
    train_x_array = train_x_df.drop(columns=["ID"]).to_numpy()
    train_y_array = train_y_df.drop(columns=["ID"]).to_numpy()
    return train_x_array, train_y_array
```


```python
data_to_train(X_train,Y_train,X_supp,139)
```




    (array([[ 0.00287013,  0.003778  ,  0.00275796, ..., -0.02263632,
             -0.01848395, -0.01659465],
            [ 0.01843633,  0.02426802,  0.01771579, ...,  0.00156197,
              0.00127544,  0.00114507],
            [-0.00874932, -0.01151685, -0.00840737, ...,  0.00800812,
              0.00653912,  0.00587074],
            ...,
            [ 0.00727536,  0.00957666,  0.00699102, ..., -0.00304767,
             -0.00248861, -0.00223424],
            [ 0.00037916,  0.00049909,  0.00036434, ..., -0.00030993,
             -0.00025307, -0.00022721],
            [ 0.01804019,  0.02374656,  0.01733513, ...,  0.00356788,
              0.00291339,  0.00261561]]),
     array([[-0.02235101],
            [ 0.00835403],
            [ 0.01221795],
            ...,
            [ 0.00347641],
            [-0.00266971],
            [-0.01650764]]))



which manifestly reproduces the above reformatting. 

# Training the model
### A brief comment on $\textrm{det} (F^TF)$

With the data now properly aranged we are in a position to begin training the model. There is one last observation in order, as we will shortly see, the determinant of $F^TF$ is incredibly small. We will solve this problem with a straightforward PCA treatment.

Recal that the solution of the linear problem is given by $\vec{\beta} = (F^T F)^{-1} F^T \vec{y}$, we thus have


```python
temp_X, temp_Y = data_to_train(X_train,Y_train,X_supp,139)

temp_beta = inv( temp_X.T @ temp_X ) @ temp_X.T @ temp_Y
temp_beta
```




    array([[ 6.68470264e+17],
           [ 9.82986828e+17],
           [-1.49521256e+18],
           [-6.02096000e+17],
           [ 4.23236079e+17],
           [-7.14196390e+17],
           [ 9.37105682e+16],
           [ 5.08860702e+17],
           [-3.53284248e+18],
           [ 1.71838954e+18],
           [-3.04631065e+18],
           [ 5.01531393e+18],
           [-4.02172257e+18],
           [-3.44549990e+18],
           [-8.63744300e+17],
           [ 8.99278511e+18],
           [-1.39737730e+18],
           [ 2.06897889e+18],
           [ 5.05524213e+17],
           [-1.75652345e+18],
           [-6.18413075e+17],
           [-7.05988700e+17],
           [ 2.27813785e+18],
           [-3.16616555e+17],
           [-1.03539086e+19],
           [ 2.39410039e+19],
           [-6.97625259e+18],
           [-1.11632354e+19],
           [ 3.48105866e+17],
           [-9.06585469e+17],
           [ 5.68642423e+17],
           [ 3.42309491e+17],
           [ 1.27806341e+19],
           [ 5.95186984e+18],
           [-2.91807923e+18],
           [-1.55688621e+19],
           [-2.17399436e+18],
           [-4.29173244e+18],
           [ 8.18043662e+18],
           [-1.04646930e+18],
           [ 1.92822515e+16],
           [-1.15102229e+18],
           [-5.91067257e+17],
           [ 2.62667910e+18],
           [ 5.27302928e+17],
           [ 3.10076637e+18],
           [-5.07797601e+18],
           [ 1.09138301e+18],
           [ 5.74581190e+17],
           [-3.96109155e+17],
           [-6.06883213e+17],
           [ 5.29147983e+17],
           [-3.75586863e+18],
           [ 1.43574997e+19],
           [-5.45330812e+18],
           [-7.67524609e+18],
           [ 5.96706939e+17],
           [-5.87230617e+17],
           [-1.32815144e+18],
           [ 1.55056869e+18],
           [ 1.08431074e+19],
           [-1.39476556e+19],
           [ 9.80860624e+18],
           [-5.10650732e+18],
           [-5.95418327e+17],
           [ 6.90550324e+17],
           [-1.32337269e+17],
           [-2.13520524e+17],
           [-3.32213496e+17],
           [ 1.00999872e+17],
           [ 4.53958761e+17],
           [-2.53524439e+17],
           [ 1.13863263e+18],
           [ 7.41658287e+17],
           [-2.94739965e+18],
           [ 8.55191498e+17],
           [-1.60889278e+18],
           [ 4.01390757e+17],
           [-2.21320978e+17],
           [ 1.34396904e+18],
           [ 1.33676741e+16],
           [-7.52041311e+17],
           [ 9.76502414e+17],
           [ 1.95845086e+17],
           [ 9.72780419e+17],
           [ 2.68216280e+16],
           [-7.09387058e+17],
           [-1.21801357e+17],
           [-3.23908821e+18],
           [ 2.74160551e+18],
           [ 4.82778532e+18],
           [-4.72662719e+18],
           [-1.78369945e+17],
           [-3.69852135e+17],
           [-1.67521170e+17],
           [ 8.61587487e+17],
           [ 1.04854639e+19],
           [-4.31629800e+18],
           [-4.69486039e+18],
           [-2.52890359e+18],
           [-1.26657375e+18],
           [ 6.32425752e+18],
           [ 9.10056910e+17],
           [-8.53414583e+18],
           [ 1.78183963e+15],
           [-4.63735590e+17],
           [-9.29952240e+16],
           [ 5.51008788e+17],
           [ 4.45032913e+18],
           [-9.22903705e+17],
           [-8.30569605e+17],
           [-2.28370487e+18],
           [ 4.97562952e+18],
           [ 2.62765461e+17],
           [ 2.86892913e+18],
           [-8.28321827e+18],
           [ 7.31515446e+17],
           [-8.85370324e+17],
           [ 6.78140671e+18],
           [-7.12035324e+18],
           [ 1.65776387e+18],
           [-2.53831277e+18],
           [ 2.39487234e+18],
           [-1.29324097e+18],
           [-1.88922803e+17],
           [ 3.17470293e+16],
           [-1.58494226e+17],
           [ 3.48731899e+17],
           [-8.50946007e+17],
           [ 3.22523812e+17],
           [ 1.36695824e+17],
           [ 2.85338696e+17],
           [-1.31947596e+17],
           [-6.01115295e+17],
           [ 2.48501484e+17],
           [ 1.04492664e+18],
           [ 1.85156904e+17],
           [-1.72852622e+17],
           [ 1.42030607e+17],
           [-1.07801639e+17],
           [-4.85455657e+16],
           [ 1.12901809e+17],
           [ 2.02452365e+15],
           [-7.62915629e+16],
           [ 9.03740337e+17],
           [-1.17558179e+18],
           [-3.58164603e+17],
           [ 1.06296579e+18],
           [ 2.11520203e+17],
           [-1.77324418e+16],
           [-2.85738480e+17],
           [ 1.10641735e+17],
           [ 2.06463951e+17],
           [-8.41344412e+16],
           [-1.66181566e+17],
           [ 6.32805223e+16],
           [-3.92318766e+17],
           [-5.91062613e+17],
           [ 1.99561166e+18],
           [-7.57190737e+17],
           [-1.47933282e+17],
           [ 1.18829376e+17],
           [ 4.15610883e+16],
           [-5.54659054e+16],
           [-2.50740831e+17],
           [-2.92966671e+18],
           [ 3.17049732e+18],
           [ 9.92685451e+15],
           [ 3.49659021e+17],
           [ 9.44302790e+14],
           [ 3.46023699e+17],
           [-7.19565996e+17],
           [ 8.57164893e+16],
           [-1.51516667e+17],
           [-9.10357774e+16],
           [ 1.95205906e+17],
           [-1.75860026e+17],
           [ 4.21439865e+17],
           [-1.23431597e+17],
           [-1.76227104e+17],
           [ 2.87445450e+17],
           [-8.72458486e+16],
           [-1.64240019e+17],
           [-2.44533842e+16],
           [ 3.23710723e+15],
           [ 8.06148583e+14],
           [ 3.38225806e+16],
           [-4.29247197e+16],
           [-2.53333120e+17],
           [ 1.30861556e+17],
           [ 2.69221093e+16],
           [ 5.79806717e+16],
           [ 1.58129926e+16],
           [ 2.73209055e+16],
           [ 9.86200304e+16],
           [-2.01784978e+17],
           [-7.12481305e+16],
           [ 1.14329854e+17],
           [-1.55067441e+17],
           [ 1.11106877e+17],
           [ 6.22268213e+15],
           [-1.85731517e+15],
           [ 6.00535810e+15],
           [-1.04263723e+16],
           [ 3.88293987e+16],
           [ 1.55919773e+16],
           [-1.04971399e+17],
           [ 2.70228555e+16],
           [-1.52561906e+16],
           [ 5.48474224e+15],
           [ 4.77585364e+15],
           [ 3.73312284e+15],
           [-4.90483193e+17],
           [ 2.36399254e+17],
           [ 3.50075586e+16],
           [ 2.03933819e+17],
           [-1.67464153e+16],
           [ 1.92527848e+16],
           [ 1.56948845e+16],
           [-2.59315314e+16],
           [-3.36032296e+17],
           [ 4.66693448e+17],
           [ 1.02772782e+16],
           [-2.07666408e+17],
           [ 6.18879404e+15],
           [-1.21776956e+16],
           [-5.73686558e+15],
           [ 1.44028351e+16],
           [ 4.26473231e+16],
           [ 4.50807891e+16],
           [ 2.65161270e+15],
           [-1.09128506e+17],
           [-3.74792294e+16],
           [ 2.23769401e+16],
           [-4.70350870e+15],
           [ 1.98181506e+16],
           [-7.78330314e+16],
           [ 1.29740459e+17],
           [-2.46449361e+15],
           [-8.64655984e+16],
           [-1.19121370e+15],
           [ 6.25345639e+15],
           [ 4.50398966e+16],
           [-5.92843313e+16],
           [ 5.30072132e+16],
           [-4.50841162e+16],
           [-3.18765215e+16],
           [ 2.81951656e+16],
           [-6.22564888e+15],
           [-2.86263873e+15],
           [ 3.20339068e+16],
           [-2.20164295e+16],
           [ 4.89581030e+16],
           [ 1.69991840e+16],
           [-3.45113364e+16],
           [-7.94292064e+16],
           [-3.23226304e+15],
           [ 1.00644725e+16],
           [-1.14662940e+15],
           [-1.11420162e+16],
           [-4.13502449e+15],
           [-5.63474159e+15],
           [ 1.57383314e+14],
           [ 1.10128197e+16],
           [ 6.01321167e+16],
           [-2.24042804e+16],
           [ 1.14687772e+17],
           [-1.55872978e+17],
           [-1.13017079e+17],
           [ 8.76624539e+16],
           [ 5.03980364e+15],
           [ 3.48021616e+16],
           [-1.04624489e+16],
           [-2.60782412e+14],
           [ 5.60421204e+15],
           [ 6.20926961e+15],
           [ 3.40936736e+15],
           [-2.06909113e+15],
           [-6.02601432e+15],
           [ 5.84748685e+15],
           [-1.94663188e+15],
           [ 5.58642533e+15],
           [ 2.61794046e+16],
           [-3.59732217e+16],
           [-9.23845936e+15],
           [ 4.65152913e+15],
           [-2.43707146e+15],
           [ 9.40680554e+15],
           [ 2.56535751e+16],
           [-6.40366197e+16],
           [ 3.89672927e+16],
           [ 1.03264873e+16],
           [ 5.70782578e+16],
           [ 1.63311000e+15],
           [ 2.73437139e+16],
           [-9.83200216e+16],
           [ 4.69400044e+16],
           [ 6.37769032e+16],
           [-8.24360881e+16],
           [-7.88458435e+16],
           [-3.06867671e+17],
           [ 3.99014342e+16],
           [ 6.95224548e+16],
           [ 1.72616623e+17],
           [ 6.27351685e+15],
           [-6.72660264e+15],
           [-5.87452025e+16],
           [ 6.44388021e+16],
           [ 8.54386375e+15],
           [-2.14862154e+15],
           [ 3.05540141e+14],
           [-5.57838522e+15],
           [ 1.49249602e+15],
           [ 3.78423383e+15],
           [-2.83940566e+15],
           [-2.26626595e+15],
           [ 1.77475628e+16],
           [-5.79465784e+15],
           [ 4.37710644e+16],
           [-6.83745387e+16],
           [-3.38919956e+15],
           [ 6.02219096e+15],
           [-4.76638753e+15],
           [-1.27890845e+15],
           [ 3.09845577e+16],
           [-1.04253267e+17],
           [ 1.02848718e+17],
           [-1.65025274e+16],
           [ 1.90937898e+16],
           [-2.36415538e+16],
           [-1.97645207e+15],
           [ 1.45818655e+16],
           [ 3.27600005e+15],
           [-1.34435872e+15],
           [-7.77637650e+15],
           [ 6.47855257e+15],
           [-1.90573551e+15],
           [-4.32485121e+15],
           [ 6.81630187e+15],
           [-4.86113768e+14],
           [ 1.09785350e+16],
           [-1.06815756e+16],
           [-4.37464517e+15],
           [ 4.63160285e+15],
           [ 9.58346031e+14],
           [-6.70753470e+15],
           [ 7.50264669e+15],
           [ 8.77988209e+14],
           [ 2.61419970e+15],
           [ 6.45143401e+16],
           [-6.71562855e+16],
           [ 2.77922632e+13],
           [ 6.06209002e+15],
           [-5.34797225e+15],
           [-7.55157212e+15],
           [ 8.86824891e+15],
           [-4.63130034e+15],
           [-4.80079832e+15],
           [ 7.47429352e+15],
           [ 2.05974217e+15],
           [ 1.85964878e+15],
           [ 6.58389862e+15],
           [-8.21341368e+15],
           [-7.13681160e+14],
           [ 1.17560503e+15],
           [-1.87390042e+15],
           [ 9.07182796e+14],
           [-8.30431342e+13],
           [ 2.39767255e+14],
           [ 2.14244904e+13],
           [-3.62957076e+13],
           [-2.46319640e+14],
           [ 8.06838996e+13],
           [ 4.62609035e+13],
           [-3.10545141e+13],
           [-1.01735837e+14],
           [-2.20848456e+13],
           [ 1.21204599e+14],
           [-1.79242302e+14],
           [ 8.20430652e+13],
           [ 5.94620829e+14],
           [-5.77351697e+14],
           [ 1.30048354e+15],
           [-1.55536842e+15],
           [ 6.96686364e+14],
           [ 3.80760898e+14],
           [-8.41340108e+14],
           [-9.21070787e+14],
           [ 3.52552767e+15],
           [-5.59837902e+15],
           [-2.47366939e+15],
           [ 5.45878277e+15],
           [-4.93316911e+14],
           [ 2.76877106e+14],
           [ 7.61451870e+14],
           [-5.21043921e+14],
           [ 8.70269722e+00],
           [ 1.26428397e+01],
           [-1.05881084e+01],
           [-1.73090098e+01]])



which is remarkable that it managed to invert it but is for all purposes and effects a singular matrix. And indeed we see that a large part of the spectrum is tiny:


```python
lambs, _ =  LA.eigh(temp_X.T @ temp_X )
plt.scatter(range(len(lambs)),lambs)
plt.ylabel("Spectrum of X^T X")
plt.title("minimum eigenval="+str(min(lambs)))
plt.axvline([300],ls="--",c="black",alpha=0.5)
plt.yscale("log")
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_36_0.png)
    


What an interesting feature this gap in the spectrum. With that cutoff at around $100$ eigenvalues (vertical line) this is screaming for PCA! 


```python
pca = PCA(n_components=0.99) # explain 99% of variability
pca.fit( temp_X.T @ temp_X )
print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))
```

    [0.6890955  0.04495872 0.02910278 0.02493956 0.02048768 0.01867304
     0.01510933 0.01301305 0.01149781 0.01015708 0.00824222 0.00783082
     0.00764701 0.00687102 0.00598082 0.00593555 0.00544787 0.00523669
     0.00496757 0.00456368 0.00435993 0.00426588 0.0040271  0.00365229
     0.00356204 0.00323537 0.00312497 0.00289106 0.00277724 0.00250244
     0.00244335 0.00212096 0.00199401 0.00189756 0.0017849  0.00158884
     0.00147037 0.0013594  0.00119562]
    39


With just `39` components we explained $+99\%$ of the data! We inspect the first principal component to see what we can learn.


```python
plt.plot(range(100),pca.components_[0][0:-3:4],alpha=0.5,label="CLASS_1")
plt.plot(range(100),pca.components_[0][1:-2:4],alpha=0.5,label="CLASS_2")
plt.plot(range(100),pca.components_[0][2:-1:4],alpha=0.5,label="CLASS_3")
plt.plot(range(100),pca.components_[0][3::4],alpha=0.5,label="CLASS_4")
plt.legend(loc="right")
plt.title("First PCA eigenvector of data for target  return 139.")
plt.ylabel("feature weight")
plt.xlabel("features")
```




    Text(0.5, 0, 'features')




    
![png](acroscarrillo_work_files/acroscarrillo_work_40_1.png)
    


First off, there seems to be no warning signs. We can also observe that weighted returns from `CLASS_2` tend to account for more variability than the rest . This makes sense, observe that it's the class closest to its average (which is $0$) so one expects on average it's weighted return to be less suppressed. This is a direct consequence our model choice which deserves some caution.


```python
X_supp[X_supp["ID_asset"] == 139]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID_asset</th>
      <th>CLASS_LEVEL_1</th>
      <th>CLASS_LEVEL_2</th>
      <th>CLASS_LEVEL_3</th>
      <th>CLASS_LEVEL_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>139</td>
      <td>-1.363192</td>
      <td>-1.183905</td>
      <td>-1.458515</td>
      <td>-1.582341</td>
    </tr>
  </tbody>
</table>
</div>



With that out of the way, let's project the data to a smaller subspace using PCA before we train our model


```python
pca = PCA(n_components=0.99) # explain 99% of variability
pca.fit( temp_X.T @ temp_X )
X_temp_PCA = pca.transform(temp_X)
temp_beta = inv( X_temp_PCA.T @ X_temp_PCA ) @ X_temp_PCA.T @ temp_Y
temp_beta.shape
```




    (39, 1)



### Training the model

The strategy is to apply PCA to the data, train it and test it individually for each target liquid return.

PCA + train/test split:


```python
# again we pick 139 as our example asset.
X_train_139, Y_train_139 = data_to_train(X_train,Y_train,X_supp,139)

pca = PCA(n_components=0.999) # explain 99.9% of variability
pca.fit( X_train_139 )
X_train_139_PCA = pca.transform(X_train_139)

x_train, x_test, y_train, y_test = train_test_split(X_train_139_PCA, Y_train_139, test_size=0.25)
```

Fit:


```python
# note this wont have an intercept term
beta_139 = inv( x_train.T @ x_train ) @ x_train.T @ y_train
beta_139.shape
```




    (94, 1)



And the test is done using our proposed $\hat{Y}_j = \vec{F_j} \cdot \vec{\beta_j}$


```python
y_pred = x_test @ beta_139
y_true = y_test
```

We will be evaluated on some custom metric so let's use it to evaluate our model:


```python
def custom_loss(y_true,y_pred):
    sign_sum = 0.5 * np.abs( np.sign(y_true) + np.sign(y_pred) )
    sum_array = np.multiply( np.abs(y_true), sign_sum ) 
    sum_term = np.sum( sum_array )
    norm_term = np.sum( np.abs(y_true) )
    return np.divide( sum_term,norm_term ) 
```


```python
custom_loss(y_true,y_pred) # test accu
```




    0.5274801914990961




```python
y_pred = x_train @ beta_139
y_true = y_train
custom_loss(y_true,y_pred) # train accu

```




    0.6369733706515371




```python
plt.scatter(range(len(beta_139)),sorted(beta_139))
plt.yscale("log")
plt.ylabel("beta coefficients")
plt.title("Largest beta = " + str(max(beta_139)[0]))
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_55_0.png)
    


These coefficients seem small enough but they can get quite large when we use a large number of principal components which might mean we need to include some form of regularisation in the future (turns out it doesnt seem to help). To address this issue, we are now going to proceed to do a more systematic model analysis. 

## Linear model analysis

It seems like with the particular model above we can get around $60\%$ test accurancy which is not too bad but not too great either. However, before we start drawing any conclusion let's squeeze this model all we can and see what we can learn. For this, we are going to wrap the above model in a single function and we are going to study the effect of tweeking the hyperparameters (number of principal components and test/train split) in detail. So here's what this function looks like:


```python
from sklearn.linear_model import Ridge

def custom_loss_no_norm(y_true,y_pred):
    sign_sum = 0.5 * np.abs( np.sign(y_true) + np.sign(y_pred) )
    sum_array = np.multiply( np.abs(y_true), sign_sum ) 
    sum_term = np.sum( sum_array )
    return sum_term 

def inneficient_linear_model_stats(X_train,Y_train,X_supp,test_percent,PCA_percent=0.99):
    ID_target_list = X_train["ID_TARGET"].unique()
    
    mse_tot_train,mse_tot_test = 0,0
    test_loss_no_norm, test_norm1_term = 0,0
    train_loss_no_norm, train_norm1_term = 0,0
    for target in ID_target_list:
        X_train_target, Y_train_target = data_to_train(X_train,Y_train,X_supp,target)
        
        pca = PCA(n_components=PCA_percent) # explain % of variability
        pca.fit( X_train_target.T @ X_train_target )
        X_train_target_PCA = pca.transform(X_train_target)

        # add y_intercept
        X_train_target_PCA = np.hstack((X_train_target_PCA, np.ones((1,X_train_target_PCA.shape[0])).T))

        x_train, x_test, y_train, y_test = train_test_split(X_train_target_PCA, Y_train_target, test_size=test_percent)

        # no regu
        beta = inv( x_train.T @ x_train ) @ x_train.T @ y_train

        y_pred = x_test @ beta
        y_true = y_test
        mse_tot_test += mean_squared_error(y_true, y_pred)
        test_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        test_norm1_term += np.sum( np.abs(y_true))

        y_pred = x_train @ beta
        y_true = y_train
        mse_tot_train += mean_squared_error(y_true, y_pred)
        train_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        train_norm1_term += np.sum( np.abs(y_true))

    test_accu = test_loss_no_norm/test_norm1_term
    train_accu = train_loss_no_norm/train_norm1_term
    return test_accu, train_accu, mse_tot_test/len(ID_target_list), mse_tot_train/len(ID_target_list)
```


```python
inneficient_linear_model_stats(X_train,Y_train,X_supp,0.2,PCA_percent=0.99)
```




    (0.7277615125100152,
     0.7557121087730975,
     0.0003785509114923225,
     0.00035816587913660344)



Now that's much better! If we are going to fine tune this model we are going to need a more efficient way of doing so. Note that every time we call `inneficient_linear_model_stats(X_train,Y_train,X_supp,test_percent,PCA_percen)` we compute the PCA of the data which amounts at doing $100$ SVDs. But if all we want to be choosing is the number of principal components used this is unecessary! Let's instead define a function that returns all the PCA objects so that we can use them later on:


```python
def data_PCA(X_train,Y_train,X_supp):
    ID_target_list = X_train["ID_TARGET"].unique()

    PCA_ls = []
    for target in ID_target_list:
        X_train_target, Y_train_target = data_to_train(X_train,Y_train,X_supp,target)
        
        pca = PCA(n_components=1) # compute all of them
        PCA_ls.append( PCA().fit( X_train_target.T @ X_train_target ) ) 

    return PCA_ls

def linear_model_stats(PCA_ls,X_train,Y_train,X_supp,test_percent,PCA_n):
    ID_target_list = X_train["ID_TARGET"].unique()
    
    mse_tot_train,mse_tot_test = 0,0
    test_loss_no_norm, test_norm1_term = 0,0
    train_loss_no_norm, train_norm1_term = 0,0
    for (j,target) in enumerate(ID_target_list):
        X_train_target, Y_train_target = data_to_train(X_train,Y_train,X_supp,target)

        pca_trnsf = PCA_ls[j].components_[0:PCA_n+1,:]

        X_train_target_PCA =  X_train_target @ pca_trnsf.T

        # add y_intercept
        X_train_target_PCA = np.hstack((X_train_target_PCA, np.ones((1,X_train_target_PCA.shape[0])).T))

        x_train, x_test, y_train, y_test = train_test_split(X_train_target_PCA, Y_train_target, test_size=test_percent)

        # no regu
        beta = inv( x_train.T @ x_train ) @ x_train.T @ y_train

        y_pred = x_test @ beta
        y_true = y_test
        mse_tot_test += mean_squared_error(y_true, y_pred)
        test_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        test_norm1_term += np.sum( np.abs(y_true))

        y_pred = x_train @ beta
        y_true = y_train
        mse_tot_train += mean_squared_error(y_true, y_pred)
        train_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        train_norm1_term += np.sum( np.abs(y_true))

    test_accu = test_loss_no_norm/test_norm1_term
    train_accu = train_loss_no_norm/train_norm1_term
    return test_accu, train_accu, mse_tot_test/len(ID_target_list), mse_tot_train/len(ID_target_list)
```

So now we have the lengthy `data_PCA(X_train,Y_train,X_supp)`:


```python
model_PCAs =  data_PCA(X_train,Y_train,X_supp)
```

But the now faster `linear_model_stats(model_PCAs,X_train,Y_train,X_supp,0.2,PCA_percent=40)` !


```python
linear_model_stats(model_PCAs,X_train,Y_train,X_supp,0.2,40)
```




    (0.735298083045162,
     0.7520840961606847,
     0.0003844688635659014,
     0.00035881801183205384)



which is 4 times faster. It is reasonable then to see what happens as we increase the principal components from $1$ to $250$ since this is simply $250\times 20s = 5000s \approx 1h30min$. Let's queue this one up and go for a walk:


```python
test_perc = 0.2
mse_tot_test_ls = []
mse_tot_train_ls = []
test_accu_ls = []
train_accu_ls = []
n_pca_comp_ls = list(range(1,250,1))
for n_pca_comp in n_pca_comp_ls:
    test_accu, train_accu, mse_tot_test, mse_tot_train = linear_model_stats(model_PCAs,X_train,Y_train,X_supp,test_perc,n_pca_comp)
    test_accu_ls.append(test_accu)
    train_accu_ls.append(train_accu)

    mse_tot_test_ls.append(mse_tot_test)
    mse_tot_train_ls.append(mse_tot_train)
    print(n_pca_comp/250, test_accu, train_accu)

```

    0.004 0.7208255103579642 0.7193600494555855
    0.008 0.7236801193556139 0.7256581645973439
    0.012 0.7223729465188438 0.7289006124556503
    0.016 0.7249914945746675 0.7306881795497859
    0.02 0.7294278104839595 0.7316079053039165
    0.024 0.7275907785254038 0.7332385490640225
    0.028 0.72703314028112 0.7348883575941791
    0.032 0.7304303363399841 0.7359917164112894
    0.036 0.7320990125170005 0.7359969908394512
    0.04 0.7328677170767652 0.7367650133284498
    0.044 0.7307109060994792 0.7374221919770276
    0.048 0.7310325205020813 0.7384176954205878
    0.052 0.727519628510892 0.7406805897351012
    0.056 0.7334157115346731 0.7402938815252886
    0.06 0.7327638082806126 0.7408700037376675
    0.064 0.729532641704095 0.7421874999001835
    0.068 0.7358082695144895 0.742265161906248
    0.072 0.729989975724733 0.7442443819443513
    0.076 0.7372189241080809 0.7424138499001434
    0.08 0.7353439356662913 0.7442192254136121
    0.084 0.7334665535574886 0.7449972570508908
    0.088 0.733755519918496 0.7473792557830208
    0.092 0.735302550952468 0.7469578822168789
    0.096 0.7372987190884768 0.7458089643043708
    0.1 0.7332810662667385 0.74814760427721
    0.104 0.7409511267535126 0.7458001273745932
    0.108 0.7371651656614717 0.7470889481838213
    0.112 0.7338850224722321 0.7476646952588794
    0.116 0.7356331885413057 0.7475255061608426
    0.12 0.7319336997067177 0.7495770089400554
    0.124 0.7368658052893371 0.7476554451823468
    0.128 0.7391264448287871 0.7466773739883756
    0.132 0.7342152506325226 0.7495503436277552
    0.136 0.7328820169441259 0.7503676366997342
    0.14 0.7343359125925335 0.7504860894618346
    0.144 0.7314495567989708 0.7524880854011948
    0.148 0.7324089837505378 0.7514489815748127
    0.152 0.7313688571091468 0.7523601227445039
    0.156 0.7283801546969145 0.7539556752593805
    0.16 0.7330397065122403 0.7535013673931985
    0.164 0.7334936957094184 0.7536157360660727
    0.168 0.7319679514027337 0.7529086607546157
    0.172 0.7327538921473125 0.7540618991060355
    0.176 0.734173830179452 0.7538804393571789
    0.18 0.7352640251828302 0.7532410687714844
    0.184 0.7343566909581679 0.7542710582169782
    0.188 0.7363048795462038 0.755127461590657
    0.192 0.733816416750265 0.7555233405515882
    0.196 0.7322513492890279 0.756660704368429
    0.2 0.7358551761152077 0.7556576897564605
    0.204 0.7291280075038415 0.7568328390583648
    0.208 0.7294401992486457 0.7569609418278211
    0.212 0.7326045472692747 0.7567866430067443
    0.216 0.7263967383601104 0.7576844976761683
    0.22 0.7289860669932642 0.7569974358045113
    0.224 0.7310584292983855 0.7575716622157641
    0.228 0.7335508668201256 0.7571612351225253
    0.232 0.7283277695549434 0.7608914469247492
    0.236 0.7292325064871884 0.7594572928258714
    0.24 0.7262216666908776 0.7600183954461593
    0.244 0.7340163080330506 0.758345658426723
    0.248 0.7298201516835648 0.7604402616000954
    0.252 0.7315406022967474 0.7596427032296498
    0.256 0.7325076113901828 0.7596925621956029
    0.26 0.7295236366275563 0.7593997831807702
    0.264 0.7308427441885654 0.7606808320547697
    0.268 0.7303240273551791 0.7616899475302928
    0.272 0.7328232705029002 0.7608293646720644
    0.276 0.7348630808930788 0.7608855729976434
    0.28 0.728873712585756 0.7624240487629003
    0.284 0.7287172343398344 0.7616176764452058
    0.288 0.7307033105917558 0.7630224979139576
    0.292 0.7351000724649893 0.7626134500880348
    0.296 0.7320486339250842 0.7627466251082863
    0.3 0.7339410571760996 0.7638399436752435
    0.304 0.7298390694969172 0.764497751807866
    0.308 0.7301502171003292 0.7636430847432241
    0.312 0.7292749881876105 0.7647824566793164
    0.316 0.732071084440798 0.764679441619391
    0.32 0.7291532112807595 0.766981886783639
    0.324 0.7363839694265458 0.7658607698691087
    0.328 0.7296335744194099 0.76644685446081
    0.332 0.7243694973091542 0.7676463468239201
    0.336 0.7306581072923387 0.7668415107680882
    0.34 0.7302578035299905 0.7672297344608613
    0.344 0.7299651145394109 0.7676478051801188
    0.348 0.7326289661753651 0.7674861413795181
    0.352 0.7341998559510192 0.767505880767927
    0.356 0.7264706896913349 0.7688802481937317
    0.36 0.7279912991321014 0.7684538624616001
    0.364 0.7338510226509918 0.7679670560071561
    0.368 0.7278438321743939 0.7699441823024391
    0.372 0.7320463532918956 0.7690089956513055
    0.376 0.7267072573005745 0.7701182560271633
    0.38 0.7309612423501186 0.7701266797938403
    0.384 0.7275993653105585 0.7692506293655537
    0.388 0.7293502300551113 0.769462999926738
    0.392 0.731789347845224 0.7704979006654077
    0.396 0.729684832903357 0.7697340955572064
    0.4 0.7249010121015143 0.7714201588888712
    0.404 0.7227945816123554 0.7725067901908869
    0.408 0.7293336038941098 0.7709987414411672
    0.412 0.7297414592143877 0.7718791882416601
    0.416 0.7234154524616596 0.7726201487276919
    0.42 0.7260824826017689 0.7714573747072851
    0.424 0.7240958834550774 0.7735025680376177
    0.428 0.7299279792709634 0.7709070589879036
    0.432 0.7248132526367775 0.7735042551503817
    0.436 0.7277698117997429 0.772269218010221
    0.44 0.7274138183236836 0.7730226294631495
    0.444 0.7286547585194584 0.7726651031076397
    0.448 0.7216154617085128 0.7730824930670416
    0.452 0.7248345553592654 0.7728057348727436
    0.456 0.7284740421332129 0.7739701367764812
    0.46 0.7264106902244302 0.7735658507152678
    0.464 0.7266082162680388 0.7735759841630645
    0.468 0.7293008537248713 0.774050098153683
    0.472 0.7268168849033847 0.7738070416683102
    0.476 0.7298510150068532 0.7730525723908508
    0.48 0.7244306491279895 0.7733709680321459
    0.484 0.7265924657654949 0.7746996385195216
    0.488 0.7206274035746439 0.774088273962801
    0.492 0.7246314837783735 0.7750320266357623
    0.496 0.7232663674560481 0.7751963397923268
    0.5 0.725945272437887 0.7750973219903171
    0.504 0.7220590248391111 0.7755370846754898
    0.508 0.7294613919329149 0.7735200218030911
    0.512 0.7249858525022762 0.7756278784929738
    0.516 0.7202643811895519 0.7757806968186443
    0.52 0.7234785903177683 0.7749620114489845
    0.524 0.7186509199627684 0.7773794812910338
    0.528 0.718359585219697 0.7769055209046141
    0.532 0.721065657668149 0.7767365738590772
    0.536 0.723025604531488 0.775798501150055
    0.54 0.7265130758243923 0.7756641315673487
    0.544 0.719486909752896 0.7763040311771247
    0.548 0.721536552123079 0.7770011406487359
    0.552 0.7239168064977246 0.7757155265003977
    0.556 0.7232944187532485 0.7774740847439031
    0.56 0.7215820526068554 0.7773474862576232
    0.564 0.7180550441814835 0.7776428584246026
    0.568 0.7236958919273481 0.7768700495220648
    0.572 0.7198324165212681 0.7766235196789666
    0.576 0.7220184894415046 0.776865019933564
    0.58 0.7205220980455386 0.7768845046945337
    0.584 0.7176458753724707 0.7781340918202954
    0.588 0.7219804036142559 0.7771651289072905
    0.592 0.7179813935961141 0.7786250765624525
    0.596 0.7160633897942268 0.7782110468128857
    0.6 0.7205313329633601 0.7777267976264963
    0.604 0.7208698093214471 0.7782213020449239
    0.608 0.7171773134192156 0.7785149045508472
    0.612 0.7178933996746192 0.7781337467471379
    0.616 0.7155485584477767 0.778919785168796
    0.62 0.7222235996057631 0.7792876338673469
    0.624 0.7165176223464 0.7790907704142241
    0.628 0.7131001126563405 0.7795410294830194
    0.632 0.7185354568838778 0.7787505050777401
    0.636 0.7143187939085611 0.7795105052431296
    0.64 0.7209267594750576 0.7787490546478311
    0.644 0.716915050868524 0.7800347117710136
    0.648 0.7133700926969583 0.7803178386994597
    0.652 0.7205173528538795 0.7799013076577348
    0.656 0.7163690913658249 0.7807732252969968
    0.66 0.7145761241416942 0.7799350969968956
    0.664 0.7155140191435345 0.7803037182826837
    0.668 0.7122817735473107 0.7820061841833483
    0.672 0.7185211071140053 0.7798528711452765
    0.676 0.7187195039596568 0.7807540903593947
    0.68 0.7141763770390076 0.7828504365990232
    0.684 0.7123625455372895 0.7825923144955836
    0.688 0.7153256881214324 0.7812468613734994
    0.692 0.7201381329509733 0.7827781500323372
    0.696 0.7127550505443982 0.7820848715306598
    0.7 0.7153441456791652 0.7818979203714477
    0.704 0.7143117004955072 0.7820483222462922
    0.708 0.7179901890937803 0.7821119602197585
    0.712 0.7160391151109492 0.7822440265446428
    0.716 0.7127965294077327 0.7837225579525011
    0.72 0.7116070362235934 0.7830004743496447
    0.724 0.7155362156483164 0.7833750577033904
    0.728 0.7098106490386262 0.783732408800267
    0.732 0.7096453064735249 0.7841391024076495
    0.736 0.7152289368264573 0.7829772306749706
    0.74 0.7181398010467875 0.7824366797159267
    0.744 0.7132779463987091 0.7844064575061424
    0.748 0.7098255234148614 0.7844767361073809
    0.752 0.7155891344573869 0.7846005949859641
    0.756 0.7172840942435281 0.7834672815662912
    0.76 0.709444221192679 0.7849419090805164
    0.764 0.7084913954497308 0.7858239049066883
    0.768 0.7141792679320715 0.7843398945225547
    0.772 0.7157045629220538 0.7849262629139346
    0.776 0.7096980342840656 0.7854493199275804
    0.78 0.7118664514883821 0.7843306237005697
    0.784 0.7143513426778036 0.7847648168740073
    0.788 0.7133795324517489 0.7856633253185871
    0.792 0.7088041865882142 0.7850451183483543
    0.796 0.7113387253736764 0.785371908756118
    0.8 0.7088067627750138 0.7865532822668848
    0.804 0.7103533232630221 0.7861962125304456
    0.808 0.7099211020592041 0.7864984482939704
    0.812 0.7060916796353374 0.7876509810621513
    0.816 0.709575208928405 0.7862970089384081
    0.82 0.710729857021483 0.7870379845572475
    0.824 0.7119142648246671 0.7847425743029445
    0.828 0.7071308275494249 0.7867292734080432
    0.832 0.7038762594515106 0.7873589288540168
    0.836 0.7088073245587413 0.7869366834450955
    0.84 0.7042775631517073 0.7877246546224509
    0.844 0.7094982703516146 0.7866411344538152
    0.848 0.7074610944536022 0.7873963325650406
    0.852 0.709921555519699 0.7875228544027428
    0.856 0.7099500060830751 0.7866919242776916
    0.86 0.7081864412661378 0.7878022545316893
    0.864 0.7078168138016211 0.7881003397927208
    0.868 0.7071452671096436 0.7885949024649539
    0.872 0.7052757810976497 0.7872981866935668
    0.876 0.7121331087929272 0.7880669512245477
    0.88 0.706498838050333 0.7886171127599355
    0.884 0.7079026260137906 0.7884661084742083
    0.888 0.704359476661319 0.7891433413474146
    0.892 0.705060040979655 0.789396390089256
    0.896 0.7067994930322898 0.7886432298610678
    0.9 0.7067957679281792 0.7892085897091892
    0.904 0.7074472400442969 0.7884498175977042
    0.908 0.7019371450831441 0.7911688966617948
    0.912 0.7119603100909981 0.7899718544042402
    0.916 0.70971950789667 0.7883727578526815
    0.92 0.7044299382137797 0.7895462418654969
    0.924 0.7067375370335574 0.7901080586212281
    0.928 0.7099375772268852 0.7900126337908984
    0.932 0.7008360714150833 0.7905312042927599
    0.936 0.7122430791036659 0.7885369432955813
    0.94 0.7046903194691283 0.7903499645264906
    0.944 0.7059684441822696 0.7911797552131724
    0.948 0.703565705566014 0.7922057952884733
    0.952 0.7057351846844994 0.7904247617270234
    0.956 0.7063397091144912 0.7913166474573969
    0.96 0.7037396177993389 0.7897026087548117
    0.964 0.7013463920733872 0.7916032529516122
    0.968 0.7045616975855624 0.7918167472133955
    0.972 0.7075113355908853 0.7915139785554649
    0.976 0.7076495769589256 0.790996914690064
    0.98 0.7036137868888852 0.7936681034919812
    0.984 0.703214842680617 0.7921002293490835
    0.988 0.7040987567030821 0.792707642330368
    0.992 0.7038912591299107 0.7909663401140126
    0.996 0.700608864154978 0.7926022186977324



```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,test_accu_ls, label="test",c="blue")
ax1.plot(n_pca_comp_ls,train_accu_ls, label="train",ls="dashed",c="blue")
ax1.set_ylabel('Metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,mse_tot_test_ls, label="test",c="green")
ax2.plot(n_pca_comp_ls,mse_tot_train_ls, label="train",ls="dashed",c="green")
ax2.set_ylabel('MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max test acc="+str(round(max(test_accu_ls),4))+" at n_comp="+str(n_pca_comp_ls[test_accu_ls.index(max(test_accu_ls))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_68_0.png)
    


Now we are talking! Let's see what's the effect of adjusting the test set size too, but lets limit the no. of pca components to $150$.


```python
percent_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
n_pca_comp_ls = list(range(1,150,1))
# mse_tot_test_ls_ls = []
# mse_tot_train_ls_ls = []
# test_accu_ls_ls = []
# train_accu_ls_ls = []
for percent in percent_ls:
    test_perc = percent
    mse_tot_test_ls = []
    mse_tot_train_ls = []
    test_accu_ls = []
    train_accu_ls = []
    for n_pca_comp in n_pca_comp_ls:
        test_accu, train_accu, mse_tot_test, mse_tot_train = linear_model_stats(model_PCAs,X_train,Y_train,X_supp,test_perc,n_pca_comp)
        test_accu_ls.append(test_accu)
        train_accu_ls.append(train_accu)
        mse_tot_test_ls.append(mse_tot_test)
        mse_tot_train_ls.append(mse_tot_train)
        # print(n_pca_comp/250, test_accu)

    test_accu_ls_ls.append(test_accu_ls)
    train_accu_ls_ls.append(train_accu_ls)
    mse_tot_test_ls_ls.append(mse_tot_test_ls)
    mse_tot_train_ls_ls.append(mse_tot_train_ls)
    print(percent/len(percent_ls))
```

    0.008333333333333333
    0.016666666666666666
    0.024999999999999998
    0.03333333333333333
    0.041666666666666664
    0.049999999999999996



```python

fig, axs = plt.subplots(3, 2, figsize=(12, 14), layout='constrained',sharex=True,sharey=True)
i=0
for ax in axs.flat:
    test_perc = percent_ls[i]
    mse_tot_test_ls = mse_tot_test_ls_ls[i]
    mse_tot_train_ls = mse_tot_train_ls_ls[i]
    test_accu_ls = test_accu_ls_ls[i]
    train_accu_ls = train_accu_ls_ls[i]
    i += 1

    ax_twin = ax.twinx()

    ax.plot(n_pca_comp_ls,test_accu_ls, label="test",c="blue")
    ax.plot(n_pca_comp_ls,train_accu_ls, label="train",ls="dashed",c="blue")
    ax.set_ylabel('Metric accurancy',c="blue")

    ax_twin.plot(n_pca_comp_ls,mse_tot_test_ls, label="test",c="green")
    ax_twin.plot(n_pca_comp_ls,mse_tot_train_ls, label="train",ls="dashed",c="green")
    ax_twin.set_ylabel('MSE',c="green")

    ax.set_xlabel("no. PCA components")
    plt.title("test split="+str(test_perc)+". Max test acc="+str(round(max(test_accu_ls),4))+" at n_comp="+str(n_pca_comp_ls[test_accu_ls.index(max(test_accu_ls))]))
    plt.legend()

plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_71_0.png)
    


This is looking promising! We can smooth out that noise by, given one test/train split percentage, taking averages over different splits of the data. In physics this would be called disorder average and in machine learning is apparently called cross validation. Let's see if we can tune the model further! We estimate the time of the following simulation to be $20s/(realisation\times component) \times 100 realisations \times 55 components = 110000s \approx 30h$! That's too long, let's optimise our `linear_model_stats` function further, it looks like loading the data over and over is unecesary and expensive, let's address this:


```python
def data_to_train_pd(X_train,Y_train,supp_df,liquid_ID):
    train_df = weight_returns(X_train,supp_df,liquid_ID)
    ID_list = train_df["ID"]
    test_df = Y_train.loc[Y_train['ID'].isin(ID_list)]    
    
    train_array = train_df.drop(columns=["ID"])
    test_array = test_df.drop(columns=["ID"])
    train_array["ID_TARGET"] = liquid_ID
    test_array["ID_TARGET"] = liquid_ID
    return train_array, test_array



df_x_train = pd.DataFrame()
df_y_train = pd.DataFrame()
for target_ID in X_train["ID_TARGET"].unique():
     df_temp_x, df_temp_y = data_to_train_pd(X_train,Y_train,X_supp,target_ID)
     df_x_train = pd.concat([df_x_train,df_temp_x])
     df_y_train = pd.concat([df_y_train,df_temp_y])
     
df_x_train[df_x_train["ID_TARGET"] == 139]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RET_216_0</th>
      <th>RET_216_1</th>
      <th>RET_216_2</th>
      <th>RET_216_3</th>
      <th>RET_238_0</th>
      <th>RET_238_1</th>
      <th>RET_238_2</th>
      <th>RET_238_3</th>
      <th>RET_45_0</th>
      <th>RET_45_1</th>
      <th>...</th>
      <th>RET_95_3</th>
      <th>RET_162_0</th>
      <th>RET_162_1</th>
      <th>RET_162_2</th>
      <th>RET_162_3</th>
      <th>RET_297_0</th>
      <th>RET_297_1</th>
      <th>RET_297_2</th>
      <th>RET_297_3</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002870</td>
      <td>0.003778</td>
      <td>0.002758</td>
      <td>0.002506</td>
      <td>0.006588</td>
      <td>0.008672</td>
      <td>0.006331</td>
      <td>0.005526</td>
      <td>0.002608</td>
      <td>0.003530</td>
      <td>...</td>
      <td>0.000129</td>
      <td>-0.001189</td>
      <td>-0.001181</td>
      <td>-0.000824</td>
      <td>-0.000705</td>
      <td>-0.020538</td>
      <td>-0.022636</td>
      <td>-0.018484</td>
      <td>-0.016595</td>
      <td>139</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.018436</td>
      <td>0.024268</td>
      <td>0.017716</td>
      <td>0.016094</td>
      <td>-0.001504</td>
      <td>-0.001980</td>
      <td>-0.001446</td>
      <td>-0.001262</td>
      <td>-0.011448</td>
      <td>-0.015497</td>
      <td>...</td>
      <td>-0.001055</td>
      <td>0.002120</td>
      <td>0.002104</td>
      <td>0.001468</td>
      <td>0.001257</td>
      <td>0.001417</td>
      <td>0.001562</td>
      <td>0.001275</td>
      <td>0.001145</td>
      <td>139</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-0.008749</td>
      <td>-0.011517</td>
      <td>-0.008407</td>
      <td>-0.007638</td>
      <td>0.005322</td>
      <td>0.007005</td>
      <td>0.005114</td>
      <td>0.004464</td>
      <td>0.026942</td>
      <td>0.036471</td>
      <td>...</td>
      <td>0.002445</td>
      <td>0.003585</td>
      <td>0.003558</td>
      <td>0.002483</td>
      <td>0.002125</td>
      <td>0.007266</td>
      <td>0.008008</td>
      <td>0.006539</td>
      <td>0.005871</td>
      <td>139</td>
    </tr>
    <tr>
      <th>300</th>
      <td>-0.023964</td>
      <td>-0.031545</td>
      <td>-0.023028</td>
      <td>-0.020920</td>
      <td>-0.002458</td>
      <td>-0.003235</td>
      <td>-0.002362</td>
      <td>-0.002062</td>
      <td>-0.004778</td>
      <td>-0.006468</td>
      <td>...</td>
      <td>-0.002560</td>
      <td>0.001672</td>
      <td>0.001660</td>
      <td>0.001158</td>
      <td>0.000991</td>
      <td>0.006732</td>
      <td>0.007419</td>
      <td>0.006058</td>
      <td>0.005439</td>
      <td>139</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.020806</td>
      <td>0.027387</td>
      <td>0.019992</td>
      <td>0.018163</td>
      <td>-0.060123</td>
      <td>-0.079141</td>
      <td>-0.057773</td>
      <td>-0.050430</td>
      <td>0.000171</td>
      <td>0.000231</td>
      <td>...</td>
      <td>-0.002598</td>
      <td>-0.000618</td>
      <td>-0.000614</td>
      <td>-0.000428</td>
      <td>-0.000366</td>
      <td>-0.012285</td>
      <td>-0.013541</td>
      <td>-0.011057</td>
      <td>-0.009927</td>
      <td>139</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266605</th>
      <td>-0.008584</td>
      <td>-0.011299</td>
      <td>-0.008248</td>
      <td>-0.007493</td>
      <td>-0.015306</td>
      <td>-0.020148</td>
      <td>-0.014708</td>
      <td>-0.012839</td>
      <td>-0.002338</td>
      <td>-0.003165</td>
      <td>...</td>
      <td>0.000410</td>
      <td>-0.001722</td>
      <td>-0.001709</td>
      <td>-0.001192</td>
      <td>-0.001021</td>
      <td>-0.001284</td>
      <td>-0.001415</td>
      <td>-0.001155</td>
      <td>-0.001037</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266704</th>
      <td>0.001172</td>
      <td>0.001543</td>
      <td>0.001126</td>
      <td>0.001023</td>
      <td>0.013406</td>
      <td>0.017647</td>
      <td>0.012882</td>
      <td>0.011245</td>
      <td>0.027601</td>
      <td>0.037364</td>
      <td>...</td>
      <td>-0.001366</td>
      <td>0.016677</td>
      <td>0.016553</td>
      <td>0.011551</td>
      <td>0.009887</td>
      <td>0.000602</td>
      <td>0.000663</td>
      <td>0.000542</td>
      <td>0.000486</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266803</th>
      <td>0.007275</td>
      <td>0.009577</td>
      <td>0.006991</td>
      <td>0.006351</td>
      <td>-0.002305</td>
      <td>-0.003034</td>
      <td>-0.002215</td>
      <td>-0.001933</td>
      <td>-0.009608</td>
      <td>-0.013006</td>
      <td>...</td>
      <td>0.000965</td>
      <td>-0.006591</td>
      <td>-0.006543</td>
      <td>-0.004565</td>
      <td>-0.003908</td>
      <td>-0.002765</td>
      <td>-0.003048</td>
      <td>-0.002489</td>
      <td>-0.002234</td>
      <td>139</td>
    </tr>
    <tr>
      <th>266902</th>
      <td>0.000379</td>
      <td>0.000499</td>
      <td>0.000364</td>
      <td>0.000331</td>
      <td>-0.004699</td>
      <td>-0.006185</td>
      <td>-0.004515</td>
      <td>-0.003941</td>
      <td>-0.028027</td>
      <td>-0.037941</td>
      <td>...</td>
      <td>-0.001861</td>
      <td>-0.004809</td>
      <td>-0.004773</td>
      <td>-0.003330</td>
      <td>-0.002851</td>
      <td>-0.000281</td>
      <td>-0.000310</td>
      <td>-0.000253</td>
      <td>-0.000227</td>
      <td>139</td>
    </tr>
    <tr>
      <th>267001</th>
      <td>0.018040</td>
      <td>0.023747</td>
      <td>0.017335</td>
      <td>0.015749</td>
      <td>-0.002337</td>
      <td>-0.003076</td>
      <td>-0.002246</td>
      <td>-0.001960</td>
      <td>-0.015134</td>
      <td>-0.020487</td>
      <td>...</td>
      <td>-0.000320</td>
      <td>-0.008560</td>
      <td>-0.008496</td>
      <td>-0.005929</td>
      <td>-0.005074</td>
      <td>0.003237</td>
      <td>0.003568</td>
      <td>0.002913</td>
      <td>0.002616</td>
      <td>139</td>
    </tr>
  </tbody>
</table>
<p>2739 rows × 401 columns</p>
</div>




```python
def fast_linear_model_stats(PCA_ls,df_x_train,df_y_train,test_percent,PCA_n):
    ID_target_list = X_train["ID_TARGET"].unique()
    
    mse_tot_train,mse_tot_test = 0,0
    test_loss_no_norm, test_norm1_term = 0,0
    train_loss_no_norm, train_norm1_term = 0,0
    for (j,target) in enumerate(ID_target_list):
        X_train_target = df_x_train[df_x_train["ID_TARGET"] == target]
        Y_train_target = df_y_train[df_x_train["ID_TARGET"] == target]
        X_train_target = X_train_target.drop(columns=["ID_TARGET"]).to_numpy()
        Y_train_target = Y_train_target.drop(columns=["ID_TARGET"]).to_numpy()

        pca_trnsf = PCA_ls[j].components_[0:PCA_n+1,:]

        X_train_target_PCA =  X_train_target @ pca_trnsf.T

        # add y_intercept
        X_train_target_PCA = np.hstack((X_train_target_PCA, np.ones((1,X_train_target_PCA.shape[0])).T))

        x_train, x_test, y_train, y_test = train_test_split(X_train_target_PCA, Y_train_target, test_size=test_percent)

        # no regu
        beta = inv( x_train.T @ x_train ) @ x_train.T @ y_train

        y_pred = x_test @ beta
        y_true = y_test
        mse_tot_test += mean_squared_error(y_true, y_pred)
        test_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        test_norm1_term += np.sum( np.abs(y_true))

        y_pred = x_train @ beta
        y_true = y_train
        mse_tot_train += mean_squared_error(y_true, y_pred)
        train_loss_no_norm += custom_loss_no_norm(y_true,y_pred)
        train_norm1_term += np.sum( np.abs(y_true))

    test_accu = test_loss_no_norm/test_norm1_term
    train_accu = train_loss_no_norm/train_norm1_term
    return test_accu, train_accu, mse_tot_test/len(ID_target_list), mse_tot_train/len(ID_target_list)
```


```python
fast_linear_model_stats(model_PCAs,df_x_train,df_y_train,0.1,40)
```




    (0.7364792198225127,
     0.7515951760328441,
     0.00037698489795563673,
     0.00036115935031399786)



Which is almost a factor of $10$ faster! Now each realisation should be $\approx 2min$ so the entire thing should be around $\approx 3h$. Armed with this new tool we can head back to our cross-validation:

We plot the average of our data:


```python
avg_test_mse = np.mean(test_mse_array,axis=1)
avg_train_mse = np.mean(train_mse_array,axis=1)
avg_test_accu = np.mean(test_accu_array,axis=1)
avg_train_accu = np.mean(train_accu_array,axis=1)

err_test_mse  = np.std(test_mse_array, axis=1) / np.sqrt(split_realisations)
err_train_mse  = np.std(train_mse_array, axis=1) / np.sqrt(split_realisations)
err_test_accu = np.std(test_accu_array, axis=1) / np.sqrt(split_realisations)
err_train_accu  = np.std(train_accu_array, axis=1) / np.sqrt(split_realisations)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,avg_test_accu, label="test",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_test_accu-err_test_accu, avg_test_accu+err_test_accu)

ax1.plot(n_pca_comp_ls,avg_train_accu, label="train",ls="dashed",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_train_accu-err_train_accu, avg_train_accu+err_train_accu)

ax1.set_ylim(0.73,0.75)

ax1.set_ylabel('Avg metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,avg_test_mse, label="test",c="green")
ax1.fill_between(n_pca_comp_ls, avg_test_mse-err_test_mse, avg_test_mse+err_test_mse)

ax2.plot(n_pca_comp_ls,avg_train_mse, label="train",ls="dashed",c="green")
ax1.fill_between(n_pca_comp_ls, avg_train_mse-err_train_mse, avg_train_mse+err_train_mse)

ax2.set_ylabel('Avg MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max avg test acc="+str(round(max(avg_test_accu),4))+" at n_comp="+str(n_pca_comp_ls[list(avg_test_accu).index(max(avg_test_accu))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_78_0.png)
    


Perhaps too small of a test set? Let's see $0.15$? Also, we probably dont need to average over $100$ realisations, look like it converges fast enough.


```python
test_perc = 0.15

split_realisations = 50
n_pca_comp_ls = list(range(15,70,1))

# test_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
# train_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
# test_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))
# train_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))

for j in range(split_realisations):
    for (k,n_pca_comp) in enumerate(n_pca_comp_ls):
        test_accu, train_accu, mse_tot_test, mse_tot_train = fast_linear_model_stats(model_PCAs,df_x_train,df_y_train,X_supp,test_perc,n_pca_comp)
        test_mse_array[k,j] = mse_tot_test
        train_mse_array[k,j] = mse_tot_train
        test_accu_array[k,j] = test_accu
        train_accu_array[k,j] = train_accu

    print(j/split_realisations)
```

    0.0
    0.02
    0.04
    0.06
    0.08
    0.1
    0.12
    0.14
    0.16
    0.18
    0.2
    0.22
    0.24
    0.26
    0.28
    0.3
    0.32
    0.34
    0.36
    0.38
    0.4
    0.42
    0.44
    0.46
    0.48
    0.5
    0.52
    0.54
    0.56
    0.58
    0.6
    0.62
    0.64
    0.66
    0.68
    0.7
    0.72
    0.74
    0.76
    0.78
    0.8
    0.82
    0.84
    0.86
    0.88
    0.9
    0.92
    0.94
    0.96
    0.98



```python
avg_test_mse = np.mean(test_mse_array,axis=1)
avg_train_mse = np.mean(train_mse_array,axis=1)
avg_test_accu = np.mean(test_accu_array,axis=1)
avg_train_accu = np.mean(train_accu_array,axis=1)

err_test_mse  = np.std(test_mse_array, axis=1) / np.sqrt(split_realisations)
err_train_mse  = np.std(train_mse_array, axis=1) / np.sqrt(split_realisations)
err_test_accu = np.std(test_accu_array, axis=1) / np.sqrt(split_realisations)
err_train_accu  = np.std(train_accu_array, axis=1) / np.sqrt(split_realisations)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,avg_test_accu, label="test",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_test_accu-err_test_accu, avg_test_accu+err_test_accu)

ax1.plot(n_pca_comp_ls,avg_train_accu, label="train",ls="dashed",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_train_accu-err_train_accu, avg_train_accu+err_train_accu)

ax1.set_ylim(0.73,0.75)

ax1.set_ylabel('Avg metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,avg_test_mse, label="test",c="green")
ax1.fill_between(n_pca_comp_ls, avg_test_mse-err_test_mse, avg_test_mse+err_test_mse)

ax2.plot(n_pca_comp_ls,avg_train_mse, label="train",ls="dashed",c="green")
ax1.fill_between(n_pca_comp_ls, avg_train_mse-err_train_mse, avg_train_mse+err_train_mse)

ax2.set_ylabel('Avg MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max avg test acc="+str(round(max(avg_test_accu),4))+" at n_comp="+str(n_pca_comp_ls[list(avg_test_accu).index(max(avg_test_accu))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_81_0.png)
    


`n_comp=19` it is!. It's clear at this stage that this model is capable of potentially reaching accurancies close to $75\%$. Naturally, the more train data we have, the larger the test accurancy. At this stage, all we can hope for is to choose a particular number of PCA components based on the above considerations and hope for the best. For the data challenge, we could grind the best accurancy out by trying all reasonable no. of components but this is not the best use of my time. Hence I will stop tunning the linear model here. It is also worth commenting that I did try out adding regularisation to the model but this didnt seem to help. Insead of diving into analysis what else I tried but didnt work better I have decided, in the sake of time, to focus the discussion on the model interpretation and its subsequent submission.

# Linear model interpretation and submission

## Submission

The beautiful thing about linear models is that is one of the few models in machine learning that offers simple ways of **actually** understanding the model decision making. To illustrate this, let's quickly see what we can learn from (by now our favourite) target return `RET_139`. For that we need to translate the above lessons into a predicting model. As mentioned, we will choose $19$ principal components for our PCA and train on the entire data on this (to see if we can squeeze a last drop). Let's construct that function and test it works with the train set


```python
def data_to_test_pd(X_test,supp_df,liquid_ID):
    test_df = weight_returns(X_test,supp_df,liquid_ID)
    test_df["ID_TARGET"] = liquid_ID
    return test_df

df_x_test_train = pd.DataFrame()
for target_ID in X_train["ID_TARGET"].unique():
     df_temp_x = data_to_test_pd(X_train,X_supp,target_ID)
     df_x_test_train = pd.concat([df_x_test_train,df_temp_x])
     
df_x_test_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_216_0</th>
      <th>RET_216_1</th>
      <th>RET_216_2</th>
      <th>RET_216_3</th>
      <th>RET_238_0</th>
      <th>RET_238_1</th>
      <th>RET_238_2</th>
      <th>RET_238_3</th>
      <th>RET_45_0</th>
      <th>...</th>
      <th>RET_95_3</th>
      <th>RET_162_0</th>
      <th>RET_162_1</th>
      <th>RET_162_2</th>
      <th>RET_162_3</th>
      <th>RET_297_0</th>
      <th>RET_297_1</th>
      <th>RET_297_2</th>
      <th>RET_297_3</th>
      <th>ID_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.002870</td>
      <td>0.003778</td>
      <td>0.002758</td>
      <td>0.002506</td>
      <td>0.006588</td>
      <td>0.008672</td>
      <td>0.006331</td>
      <td>0.005526</td>
      <td>0.002608</td>
      <td>...</td>
      <td>0.000129</td>
      <td>-0.001189</td>
      <td>-0.001181</td>
      <td>-0.000824</td>
      <td>-0.000705</td>
      <td>-0.020538</td>
      <td>-0.022636</td>
      <td>-0.018484</td>
      <td>-0.016595</td>
      <td>139</td>
    </tr>
    <tr>
      <th>100</th>
      <td>100</td>
      <td>0.018436</td>
      <td>0.024268</td>
      <td>0.017716</td>
      <td>0.016094</td>
      <td>-0.001504</td>
      <td>-0.001980</td>
      <td>-0.001446</td>
      <td>-0.001262</td>
      <td>-0.011448</td>
      <td>...</td>
      <td>-0.001055</td>
      <td>0.002120</td>
      <td>0.002104</td>
      <td>0.001468</td>
      <td>0.001257</td>
      <td>0.001417</td>
      <td>0.001562</td>
      <td>0.001275</td>
      <td>0.001145</td>
      <td>139</td>
    </tr>
    <tr>
      <th>200</th>
      <td>200</td>
      <td>-0.008749</td>
      <td>-0.011517</td>
      <td>-0.008407</td>
      <td>-0.007638</td>
      <td>0.005322</td>
      <td>0.007005</td>
      <td>0.005114</td>
      <td>0.004464</td>
      <td>0.026942</td>
      <td>...</td>
      <td>0.002445</td>
      <td>0.003585</td>
      <td>0.003558</td>
      <td>0.002483</td>
      <td>0.002125</td>
      <td>0.007266</td>
      <td>0.008008</td>
      <td>0.006539</td>
      <td>0.005871</td>
      <td>139</td>
    </tr>
    <tr>
      <th>300</th>
      <td>300</td>
      <td>-0.023964</td>
      <td>-0.031545</td>
      <td>-0.023028</td>
      <td>-0.020920</td>
      <td>-0.002458</td>
      <td>-0.003235</td>
      <td>-0.002362</td>
      <td>-0.002062</td>
      <td>-0.004778</td>
      <td>...</td>
      <td>-0.002560</td>
      <td>0.001672</td>
      <td>0.001660</td>
      <td>0.001158</td>
      <td>0.000991</td>
      <td>0.006732</td>
      <td>0.007419</td>
      <td>0.006058</td>
      <td>0.005439</td>
      <td>139</td>
    </tr>
    <tr>
      <th>400</th>
      <td>400</td>
      <td>0.020806</td>
      <td>0.027387</td>
      <td>0.019992</td>
      <td>0.018163</td>
      <td>-0.060123</td>
      <td>-0.079141</td>
      <td>-0.057773</td>
      <td>-0.050430</td>
      <td>0.000171</td>
      <td>...</td>
      <td>-0.002598</td>
      <td>-0.000618</td>
      <td>-0.000614</td>
      <td>-0.000428</td>
      <td>-0.000366</td>
      <td>-0.012285</td>
      <td>-0.013541</td>
      <td>-0.011057</td>
      <td>-0.009927</td>
      <td>139</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266703</th>
      <td>266703</td>
      <td>-0.012034</td>
      <td>-0.012034</td>
      <td>-0.010992</td>
      <td>-0.011348</td>
      <td>-0.021459</td>
      <td>-0.021459</td>
      <td>-0.019601</td>
      <td>-0.019828</td>
      <td>-0.004046</td>
      <td>...</td>
      <td>0.000581</td>
      <td>-0.003203</td>
      <td>-0.002199</td>
      <td>-0.001680</td>
      <td>-0.001651</td>
      <td>-0.002273</td>
      <td>-0.001773</td>
      <td>-0.001574</td>
      <td>-0.001608</td>
      <td>155</td>
    </tr>
    <tr>
      <th>266802</th>
      <td>266802</td>
      <td>0.001643</td>
      <td>0.001643</td>
      <td>0.001501</td>
      <td>0.001549</td>
      <td>0.018795</td>
      <td>0.018795</td>
      <td>0.017167</td>
      <td>0.017367</td>
      <td>0.047766</td>
      <td>...</td>
      <td>-0.001937</td>
      <td>0.031023</td>
      <td>0.021304</td>
      <td>0.016272</td>
      <td>0.015997</td>
      <td>0.001066</td>
      <td>0.000831</td>
      <td>0.000738</td>
      <td>0.000754</td>
      <td>155</td>
    </tr>
    <tr>
      <th>266901</th>
      <td>266901</td>
      <td>0.010200</td>
      <td>0.010200</td>
      <td>0.009317</td>
      <td>0.009619</td>
      <td>-0.003231</td>
      <td>-0.003231</td>
      <td>-0.002952</td>
      <td>-0.002986</td>
      <td>-0.016627</td>
      <td>...</td>
      <td>0.001368</td>
      <td>-0.012262</td>
      <td>-0.008420</td>
      <td>-0.006431</td>
      <td>-0.006322</td>
      <td>-0.004896</td>
      <td>-0.003819</td>
      <td>-0.003391</td>
      <td>-0.003464</td>
      <td>155</td>
    </tr>
    <tr>
      <th>267000</th>
      <td>267000</td>
      <td>0.000532</td>
      <td>0.000532</td>
      <td>0.000486</td>
      <td>0.000501</td>
      <td>-0.006588</td>
      <td>-0.006588</td>
      <td>-0.006017</td>
      <td>-0.006087</td>
      <td>-0.048504</td>
      <td>...</td>
      <td>-0.002639</td>
      <td>-0.008945</td>
      <td>-0.006143</td>
      <td>-0.004692</td>
      <td>-0.004612</td>
      <td>-0.000498</td>
      <td>-0.000388</td>
      <td>-0.000345</td>
      <td>-0.000352</td>
      <td>155</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>267099</td>
      <td>0.025293</td>
      <td>0.025293</td>
      <td>0.023102</td>
      <td>0.023851</td>
      <td>-0.003277</td>
      <td>-0.003277</td>
      <td>-0.002993</td>
      <td>-0.003028</td>
      <td>-0.026191</td>
      <td>...</td>
      <td>-0.000453</td>
      <td>-0.015923</td>
      <td>-0.010935</td>
      <td>-0.008352</td>
      <td>-0.008211</td>
      <td>0.005732</td>
      <td>0.004471</td>
      <td>0.003970</td>
      <td>0.004056</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 402 columns</p>
</div>




```python
def linear_model_predict(PCA_ls,df_x_test,df_x_train,df_y_train,supp_df,PCA_n=19):
    ID_target_list = df_x_test["ID_TARGET"].unique()

    prediction_df = pd.DataFrame()
    for (j,target) in enumerate(ID_target_list):
        X_train_target = df_x_train[df_x_train["ID_TARGET"] == target]
        X_train_target = X_train_target.drop(columns=["ID_TARGET"]).to_numpy()

        X_test_target = df_x_test[df_x_test["ID_TARGET"] == target]
        X_test_target = X_test_target.drop(columns=["ID_TARGET"])
        IDs = X_test_target["ID"]
        X_test_target = X_test_target.drop(columns=["ID"]).to_numpy()

        Y_train_target = df_y_train[df_x_train["ID_TARGET"] == target]
        Y_train_target = Y_train_target.drop(columns=["ID_TARGET"]).to_numpy()

        pca_trnsf = PCA_ls[j].components_[0:PCA_n+1,:]

        # transform train
        X_train_target_PCA =  X_train_target @ pca_trnsf.T
        # transform test
        X_test_target_PCA =  X_test_target @ pca_trnsf.T


        # add y_intercept train 
        X_train_target_PCA = np.hstack((X_train_target_PCA, np.ones((1,X_train_target_PCA.shape[0])).T))

        # add y_intercept test
        X_test_target_PCA = np.hstack((X_test_target_PCA, np.ones((1,X_test_target_PCA.shape[0])).T))

        # no regu
        beta = inv( X_train_target_PCA.T @ X_train_target_PCA ) @ X_train_target_PCA.T @ Y_train_target


        y_pred = X_test_target_PCA @ beta

        y_pred_df = pd.DataFrame()
        y_pred_df["ID"] = IDs
        y_pred_df["RET_TARGET"] = np.sign(y_pred)
        prediction_df = pd.concat([prediction_df,y_pred_df])
    
    return prediction_df.sort_values(by=['ID'],ignore_index=True)

predicted_df = linear_model_predict(model_PCAs,df_x_test_train,df_x_train,df_y_train,X_supp,PCA_n=19)

custom_loss(Y_train["RET_TARGET"],predicted_df["RET_TARGET"])
```




    0.7436186405690427



Which does very well as it is in fact its own training accurancy. Now for the actual test set:


```python
df_x_test = pd.DataFrame()
for target_ID in X_test["ID_TARGET"].unique():
     df_temp_x = data_to_test_pd(X_test,X_supp,target_ID)
     df_x_test = pd.concat([df_x_test,df_temp_x])
```


```python
PCA_n = 21
predicted_df = linear_model_predict(model_PCAs,df_x_test,df_x_train,df_y_train,X_supp,PCA_n)
predicted_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267101</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267102</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267103</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267104</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114463</th>
      <td>381563</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114464</th>
      <td>381564</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114465</th>
      <td>381565</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114466</th>
      <td>381566</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>114467</th>
      <td>381567</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>114468 rows × 2 columns</p>
</div>



Let's save our prediction:


```python
predicted_df.to_csv("./submission_PCA_" + str(PCA_n) + ".csv",index=False)
```

Alright, `PCA_n=19` did well, $\sim 73.9\% $. Let's try a few more to see if we can get in the $74\%$'s! Let's run a final overnight test to see if we can get a more educated guess than `PCA_n=19`:


```python
test_perc = 0.01

split_realisations = 2000
n_pca_comp_ls = list(range(17,29,1))

test_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
train_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
test_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))
train_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))

for j in range(split_realisations):
    for (k,n_pca_comp) in enumerate(n_pca_comp_ls):
        test_accu, train_accu, mse_tot_test, mse_tot_train = fast_linear_model_stats(model_PCAs,df_x_train,df_y_train,X_supp,test_perc,n_pca_comp)
        test_mse_array[k,j] = mse_tot_test
        train_mse_array[k,j] = mse_tot_train
        test_accu_array[k,j] = test_accu
        train_accu_array[k,j] = train_accu

    print(j/split_realisations)
```

    0.0
    0.0005
    0.001
    0.0015
    0.002
    0.0025
    0.003
    0.0035
    0.004
    0.0045
    0.005
    0.0055
    0.006
    0.0065
    0.007
    0.0075
    0.008
    0.0085
    0.009
    0.0095
    0.01
    0.0105
    0.011
    0.0115
    0.012
    0.0125
    0.013
    0.0135
    0.014
    0.0145
    0.015
    0.0155
    0.016
    0.0165
    0.017
    0.0175
    0.018
    0.0185
    0.019
    0.0195
    0.02
    0.0205
    0.021
    0.0215
    0.022
    0.0225
    0.023
    0.0235
    0.024
    0.0245
    0.025
    0.0255
    0.026
    0.0265
    0.027
    0.0275
    0.028
    0.0285
    0.029
    0.0295
    0.03
    0.0305
    0.031
    0.0315
    0.032
    0.0325
    0.033
    0.0335
    0.034
    0.0345
    0.035
    0.0355
    0.036
    0.0365
    0.037
    0.0375
    0.038
    0.0385
    0.039
    0.0395
    0.04
    0.0405
    0.041
    0.0415
    0.042
    0.0425
    0.043
    0.0435
    0.044
    0.0445
    0.045
    0.0455
    0.046
    0.0465
    0.047
    0.0475
    0.048
    0.0485
    0.049
    0.0495
    0.05
    0.0505
    0.051
    0.0515
    0.052
    0.0525
    0.053
    0.0535
    0.054
    0.0545
    0.055
    0.0555
    0.056
    0.0565
    0.057
    0.0575
    0.058
    0.0585
    0.059
    0.0595
    0.06
    0.0605
    0.061
    0.0615
    0.062
    0.0625
    0.063
    0.0635
    0.064
    0.0645
    0.065
    0.0655
    0.066
    0.0665
    0.067
    0.0675
    0.068
    0.0685
    0.069
    0.0695
    0.07
    0.0705
    0.071
    0.0715
    0.072
    0.0725
    0.073
    0.0735
    0.074
    0.0745
    0.075
    0.0755
    0.076
    0.0765
    0.077
    0.0775
    0.078
    0.0785
    0.079
    0.0795
    0.08
    0.0805
    0.081
    0.0815
    0.082
    0.0825
    0.083
    0.0835
    0.084
    0.0845
    0.085
    0.0855
    0.086
    0.0865
    0.087
    0.0875
    0.088
    0.0885
    0.089
    0.0895
    0.09
    0.0905
    0.091
    0.0915
    0.092
    0.0925
    0.093
    0.0935
    0.094
    0.0945
    0.095
    0.0955
    0.096
    0.0965
    0.097
    0.0975
    0.098
    0.0985
    0.099
    0.0995
    0.1
    0.1005
    0.101
    0.1015
    0.102
    0.1025
    0.103
    0.1035
    0.104
    0.1045
    0.105
    0.1055
    0.106
    0.1065
    0.107
    0.1075
    0.108
    0.1085
    0.109
    0.1095
    0.11
    0.1105
    0.111
    0.1115
    0.112
    0.1125
    0.113
    0.1135
    0.114
    0.1145
    0.115
    0.1155
    0.116
    0.1165
    0.117
    0.1175
    0.118
    0.1185
    0.119
    0.1195
    0.12
    0.1205
    0.121
    0.1215
    0.122
    0.1225
    0.123
    0.1235
    0.124
    0.1245
    0.125
    0.1255
    0.126
    0.1265
    0.127
    0.1275
    0.128
    0.1285
    0.129
    0.1295
    0.13
    0.1305
    0.131
    0.1315
    0.132
    0.1325
    0.133
    0.1335
    0.134
    0.1345
    0.135
    0.1355
    0.136
    0.1365
    0.137
    0.1375
    0.138
    0.1385
    0.139
    0.1395
    0.14
    0.1405
    0.141
    0.1415
    0.142
    0.1425
    0.143
    0.1435
    0.144
    0.1445
    0.145
    0.1455
    0.146
    0.1465
    0.147
    0.1475
    0.148
    0.1485
    0.149
    0.1495
    0.15
    0.1505
    0.151
    0.1515
    0.152
    0.1525
    0.153
    0.1535
    0.154
    0.1545
    0.155
    0.1555
    0.156
    0.1565
    0.157
    0.1575
    0.158
    0.1585
    0.159
    0.1595
    0.16
    0.1605
    0.161
    0.1615
    0.162
    0.1625
    0.163
    0.1635
    0.164
    0.1645
    0.165
    0.1655
    0.166
    0.1665
    0.167
    0.1675
    0.168
    0.1685
    0.169
    0.1695
    0.17
    0.1705
    0.171
    0.1715
    0.172
    0.1725
    0.173
    0.1735
    0.174
    0.1745
    0.175
    0.1755
    0.176
    0.1765
    0.177
    0.1775
    0.178
    0.1785
    0.179
    0.1795
    0.18
    0.1805
    0.181
    0.1815
    0.182
    0.1825
    0.183
    0.1835
    0.184
    0.1845
    0.185
    0.1855
    0.186
    0.1865
    0.187
    0.1875
    0.188
    0.1885
    0.189
    0.1895
    0.19
    0.1905
    0.191
    0.1915
    0.192
    0.1925
    0.193
    0.1935
    0.194
    0.1945
    0.195
    0.1955
    0.196
    0.1965
    0.197
    0.1975
    0.198
    0.1985
    0.199
    0.1995
    0.2
    0.2005
    0.201
    0.2015
    0.202
    0.2025
    0.203
    0.2035
    0.204
    0.2045
    0.205
    0.2055
    0.206
    0.2065
    0.207
    0.2075
    0.208
    0.2085
    0.209
    0.2095
    0.21
    0.2105
    0.211
    0.2115
    0.212
    0.2125
    0.213
    0.2135
    0.214
    0.2145
    0.215
    0.2155
    0.216
    0.2165
    0.217
    0.2175
    0.218
    0.2185
    0.219
    0.2195
    0.22
    0.2205
    0.221
    0.2215
    0.222
    0.2225
    0.223
    0.2235
    0.224
    0.2245
    0.225
    0.2255
    0.226
    0.2265
    0.227
    0.2275
    0.228
    0.2285
    0.229
    0.2295
    0.23
    0.2305
    0.231
    0.2315
    0.232
    0.2325
    0.233
    0.2335
    0.234
    0.2345
    0.235
    0.2355
    0.236
    0.2365
    0.237
    0.2375
    0.238
    0.2385
    0.239
    0.2395
    0.24
    0.2405
    0.241
    0.2415
    0.242
    0.2425
    0.243
    0.2435
    0.244
    0.2445
    0.245
    0.2455
    0.246
    0.2465
    0.247
    0.2475
    0.248
    0.2485
    0.249
    0.2495
    0.25
    0.2505
    0.251
    0.2515
    0.252
    0.2525
    0.253
    0.2535
    0.254
    0.2545
    0.255
    0.2555
    0.256
    0.2565
    0.257
    0.2575
    0.258
    0.2585
    0.259
    0.2595
    0.26
    0.2605
    0.261
    0.2615
    0.262
    0.2625
    0.263
    0.2635
    0.264
    0.2645
    0.265
    0.2655
    0.266
    0.2665
    0.267
    0.2675
    0.268
    0.2685
    0.269
    0.2695
    0.27
    0.2705
    0.271
    0.2715
    0.272
    0.2725
    0.273
    0.2735
    0.274
    0.2745
    0.275
    0.2755
    0.276
    0.2765
    0.277
    0.2775
    0.278
    0.2785
    0.279
    0.2795
    0.28
    0.2805
    0.281
    0.2815
    0.282
    0.2825
    0.283
    0.2835
    0.284
    0.2845
    0.285
    0.2855
    0.286
    0.2865
    0.287
    0.2875
    0.288
    0.2885
    0.289
    0.2895
    0.29
    0.2905
    0.291
    0.2915
    0.292
    0.2925
    0.293
    0.2935
    0.294
    0.2945
    0.295
    0.2955
    0.296
    0.2965
    0.297
    0.2975
    0.298
    0.2985
    0.299
    0.2995
    0.3
    0.3005
    0.301
    0.3015
    0.302
    0.3025
    0.303
    0.3035
    0.304
    0.3045
    0.305
    0.3055
    0.306
    0.3065
    0.307
    0.3075
    0.308
    0.3085
    0.309
    0.3095
    0.31
    0.3105
    0.311
    0.3115
    0.312
    0.3125
    0.313
    0.3135
    0.314
    0.3145
    0.315
    0.3155
    0.316
    0.3165
    0.317
    0.3175
    0.318
    0.3185
    0.319
    0.3195
    0.32
    0.3205
    0.321
    0.3215
    0.322
    0.3225
    0.323
    0.3235
    0.324
    0.3245
    0.325
    0.3255
    0.326
    0.3265
    0.327
    0.3275
    0.328
    0.3285
    0.329
    0.3295
    0.33
    0.3305
    0.331
    0.3315
    0.332
    0.3325
    0.333
    0.3335
    0.334
    0.3345
    0.335
    0.3355
    0.336
    0.3365
    0.337
    0.3375
    0.338
    0.3385
    0.339
    0.3395
    0.34
    0.3405
    0.341
    0.3415
    0.342
    0.3425
    0.343
    0.3435
    0.344
    0.3445
    0.345
    0.3455
    0.346
    0.3465
    0.347
    0.3475
    0.348
    0.3485
    0.349
    0.3495
    0.35
    0.3505
    0.351
    0.3515
    0.352
    0.3525
    0.353
    0.3535
    0.354
    0.3545
    0.355
    0.3555
    0.356
    0.3565
    0.357
    0.3575
    0.358
    0.3585
    0.359
    0.3595
    0.36
    0.3605
    0.361
    0.3615
    0.362
    0.3625
    0.363
    0.3635
    0.364
    0.3645
    0.365
    0.3655
    0.366
    0.3665
    0.367
    0.3675
    0.368
    0.3685
    0.369
    0.3695
    0.37
    0.3705
    0.371
    0.3715
    0.372
    0.3725
    0.373
    0.3735
    0.374
    0.3745
    0.375
    0.3755
    0.376
    0.3765
    0.377
    0.3775
    0.378
    0.3785
    0.379
    0.3795
    0.38
    0.3805
    0.381
    0.3815
    0.382
    0.3825
    0.383
    0.3835
    0.384
    0.3845
    0.385
    0.3855
    0.386
    0.3865
    0.387
    0.3875
    0.388
    0.3885
    0.389
    0.3895
    0.39
    0.3905
    0.391
    0.3915
    0.392
    0.3925
    0.393
    0.3935
    0.394
    0.3945
    0.395
    0.3955
    0.396
    0.3965
    0.397
    0.3975
    0.398
    0.3985
    0.399
    0.3995
    0.4
    0.4005
    0.401
    0.4015
    0.402
    0.4025
    0.403
    0.4035
    0.404
    0.4045
    0.405
    0.4055
    0.406
    0.4065
    0.407
    0.4075
    0.408
    0.4085
    0.409
    0.4095
    0.41
    0.4105
    0.411
    0.4115
    0.412
    0.4125
    0.413
    0.4135
    0.414
    0.4145
    0.415
    0.4155
    0.416
    0.4165
    0.417
    0.4175
    0.418
    0.4185
    0.419
    0.4195
    0.42
    0.4205
    0.421
    0.4215
    0.422
    0.4225
    0.423
    0.4235
    0.424
    0.4245
    0.425
    0.4255
    0.426
    0.4265
    0.427
    0.4275
    0.428
    0.4285
    0.429
    0.4295
    0.43
    0.4305
    0.431
    0.4315
    0.432
    0.4325
    0.433
    0.4335
    0.434
    0.4345
    0.435
    0.4355
    0.436
    0.4365
    0.437
    0.4375
    0.438
    0.4385
    0.439
    0.4395
    0.44
    0.4405
    0.441
    0.4415
    0.442
    0.4425
    0.443
    0.4435
    0.444
    0.4445
    0.445
    0.4455
    0.446
    0.4465
    0.447
    0.4475
    0.448
    0.4485
    0.449
    0.4495
    0.45
    0.4505
    0.451
    0.4515
    0.452
    0.4525
    0.453
    0.4535
    0.454
    0.4545
    0.455
    0.4555
    0.456
    0.4565
    0.457
    0.4575
    0.458
    0.4585
    0.459
    0.4595
    0.46
    0.4605
    0.461
    0.4615
    0.462
    0.4625
    0.463
    0.4635
    0.464
    0.4645
    0.465
    0.4655
    0.466
    0.4665
    0.467
    0.4675
    0.468
    0.4685
    0.469
    0.4695
    0.47
    0.4705
    0.471
    0.4715
    0.472
    0.4725
    0.473
    0.4735
    0.474
    0.4745
    0.475
    0.4755
    0.476
    0.4765
    0.477
    0.4775
    0.478
    0.4785
    0.479
    0.4795
    0.48
    0.4805
    0.481
    0.4815
    0.482
    0.4825
    0.483
    0.4835
    0.484
    0.4845
    0.485
    0.4855
    0.486
    0.4865
    0.487
    0.4875
    0.488
    0.4885
    0.489
    0.4895
    0.49
    0.4905
    0.491
    0.4915
    0.492
    0.4925
    0.493
    0.4935
    0.494
    0.4945
    0.495
    0.4955
    0.496
    0.4965
    0.497
    0.4975
    0.498
    0.4985
    0.499
    0.4995
    0.5
    0.5005
    0.501
    0.5015
    0.502
    0.5025
    0.503
    0.5035
    0.504
    0.5045
    0.505
    0.5055
    0.506
    0.5065
    0.507
    0.5075
    0.508
    0.5085
    0.509
    0.5095
    0.51
    0.5105
    0.511
    0.5115
    0.512
    0.5125
    0.513
    0.5135
    0.514
    0.5145
    0.515
    0.5155
    0.516
    0.5165
    0.517
    0.5175
    0.518
    0.5185
    0.519
    0.5195
    0.52
    0.5205
    0.521
    0.5215
    0.522
    0.5225
    0.523
    0.5235
    0.524
    0.5245
    0.525
    0.5255
    0.526
    0.5265
    0.527
    0.5275
    0.528
    0.5285
    0.529
    0.5295
    0.53
    0.5305
    0.531
    0.5315
    0.532
    0.5325
    0.533
    0.5335
    0.534
    0.5345
    0.535
    0.5355
    0.536
    0.5365
    0.537
    0.5375
    0.538
    0.5385
    0.539
    0.5395
    0.54
    0.5405
    0.541
    0.5415
    0.542
    0.5425
    0.543
    0.5435
    0.544
    0.5445
    0.545
    0.5455
    0.546
    0.5465
    0.547
    0.5475
    0.548
    0.5485
    0.549
    0.5495
    0.55
    0.5505
    0.551
    0.5515
    0.552
    0.5525
    0.553
    0.5535
    0.554
    0.5545
    0.555
    0.5555
    0.556
    0.5565
    0.557
    0.5575
    0.558
    0.5585
    0.559
    0.5595
    0.56
    0.5605
    0.561
    0.5615
    0.562
    0.5625
    0.563
    0.5635
    0.564
    0.5645
    0.565
    0.5655
    0.566
    0.5665
    0.567
    0.5675
    0.568
    0.5685
    0.569
    0.5695
    0.57
    0.5705
    0.571
    0.5715
    0.572
    0.5725
    0.573
    0.5735
    0.574
    0.5745
    0.575
    0.5755
    0.576
    0.5765
    0.577
    0.5775
    0.578
    0.5785
    0.579
    0.5795
    0.58
    0.5805
    0.581
    0.5815
    0.582
    0.5825
    0.583
    0.5835
    0.584
    0.5845
    0.585
    0.5855
    0.586
    0.5865
    0.587
    0.5875
    0.588
    0.5885
    0.589
    0.5895
    0.59
    0.5905
    0.591
    0.5915
    0.592
    0.5925
    0.593
    0.5935
    0.594
    0.5945
    0.595
    0.5955
    0.596
    0.5965
    0.597
    0.5975
    0.598
    0.5985
    0.599
    0.5995
    0.6
    0.6005
    0.601
    0.6015
    0.602
    0.6025
    0.603
    0.6035
    0.604
    0.6045
    0.605
    0.6055
    0.606
    0.6065
    0.607
    0.6075
    0.608
    0.6085
    0.609
    0.6095
    0.61
    0.6105
    0.611
    0.6115
    0.612
    0.6125
    0.613
    0.6135
    0.614
    0.6145
    0.615
    0.6155
    0.616
    0.6165
    0.617
    0.6175
    0.618
    0.6185
    0.619
    0.6195
    0.62
    0.6205
    0.621
    0.6215
    0.622
    0.6225
    0.623
    0.6235
    0.624
    0.6245
    0.625
    0.6255
    0.626
    0.6265
    0.627
    0.6275
    0.628
    0.6285
    0.629
    0.6295
    0.63
    0.6305
    0.631
    0.6315
    0.632
    0.6325
    0.633
    0.6335
    0.634
    0.6345
    0.635
    0.6355
    0.636
    0.6365
    0.637
    0.6375
    0.638
    0.6385
    0.639
    0.6395
    0.64
    0.6405
    0.641
    0.6415
    0.642
    0.6425
    0.643
    0.6435
    0.644
    0.6445
    0.645
    0.6455
    0.646
    0.6465
    0.647
    0.6475
    0.648
    0.6485
    0.649
    0.6495
    0.65
    0.6505
    0.651
    0.6515
    0.652
    0.6525
    0.653
    0.6535
    0.654
    0.6545
    0.655
    0.6555
    0.656
    0.6565
    0.657
    0.6575
    0.658
    0.6585
    0.659
    0.6595
    0.66
    0.6605
    0.661
    0.6615
    0.662
    0.6625
    0.663
    0.6635
    0.664
    0.6645
    0.665
    0.6655
    0.666
    0.6665
    0.667
    0.6675
    0.668
    0.6685
    0.669
    0.6695
    0.67
    0.6705
    0.671
    0.6715
    0.672
    0.6725
    0.673
    0.6735
    0.674
    0.6745
    0.675
    0.6755
    0.676
    0.6765
    0.677
    0.6775
    0.678
    0.6785
    0.679
    0.6795
    0.68
    0.6805
    0.681
    0.6815
    0.682
    0.6825
    0.683
    0.6835
    0.684
    0.6845
    0.685
    0.6855
    0.686
    0.6865
    0.687
    0.6875
    0.688
    0.6885
    0.689
    0.6895
    0.69
    0.6905
    0.691
    0.6915
    0.692
    0.6925
    0.693
    0.6935
    0.694
    0.6945
    0.695
    0.6955
    0.696
    0.6965
    0.697
    0.6975
    0.698
    0.6985
    0.699
    0.6995
    0.7
    0.7005
    0.701
    0.7015
    0.702
    0.7025
    0.703
    0.7035
    0.704
    0.7045
    0.705
    0.7055
    0.706
    0.7065
    0.707
    0.7075
    0.708
    0.7085
    0.709
    0.7095
    0.71
    0.7105
    0.711
    0.7115
    0.712
    0.7125
    0.713
    0.7135
    0.714
    0.7145
    0.715
    0.7155
    0.716
    0.7165
    0.717
    0.7175
    0.718
    0.7185
    0.719
    0.7195
    0.72
    0.7205
    0.721
    0.7215
    0.722
    0.7225
    0.723
    0.7235
    0.724
    0.7245
    0.725
    0.7255
    0.726
    0.7265
    0.727
    0.7275
    0.728
    0.7285
    0.729
    0.7295
    0.73
    0.7305
    0.731
    0.7315
    0.732
    0.7325
    0.733
    0.7335
    0.734
    0.7345
    0.735
    0.7355
    0.736
    0.7365
    0.737
    0.7375
    0.738
    0.7385
    0.739
    0.7395
    0.74
    0.7405
    0.741
    0.7415
    0.742
    0.7425
    0.743
    0.7435
    0.744
    0.7445
    0.745
    0.7455
    0.746
    0.7465
    0.747
    0.7475
    0.748
    0.7485
    0.749
    0.7495
    0.75
    0.7505
    0.751
    0.7515
    0.752
    0.7525
    0.753
    0.7535
    0.754
    0.7545
    0.755
    0.7555
    0.756
    0.7565
    0.757
    0.7575
    0.758
    0.7585
    0.759
    0.7595
    0.76
    0.7605
    0.761
    0.7615
    0.762
    0.7625
    0.763
    0.7635
    0.764
    0.7645
    0.765
    0.7655
    0.766
    0.7665
    0.767
    0.7675
    0.768
    0.7685
    0.769
    0.7695
    0.77
    0.7705
    0.771
    0.7715
    0.772
    0.7725
    0.773
    0.7735
    0.774
    0.7745
    0.775
    0.7755
    0.776
    0.7765
    0.777
    0.7775
    0.778
    0.7785
    0.779
    0.7795
    0.78
    0.7805
    0.781
    0.7815
    0.782
    0.7825
    0.783
    0.7835
    0.784
    0.7845
    0.785
    0.7855
    0.786
    0.7865
    0.787
    0.7875
    0.788
    0.7885
    0.789
    0.7895
    0.79
    0.7905
    0.791
    0.7915
    0.792
    0.7925
    0.793
    0.7935
    0.794
    0.7945
    0.795
    0.7955
    0.796
    0.7965
    0.797
    0.7975
    0.798
    0.7985
    0.799
    0.7995
    0.8
    0.8005
    0.801
    0.8015
    0.802
    0.8025
    0.803
    0.8035
    0.804
    0.8045
    0.805
    0.8055
    0.806
    0.8065
    0.807
    0.8075
    0.808
    0.8085
    0.809
    0.8095
    0.81
    0.8105
    0.811
    0.8115
    0.812
    0.8125
    0.813
    0.8135
    0.814
    0.8145
    0.815
    0.8155
    0.816
    0.8165
    0.817
    0.8175
    0.818
    0.8185
    0.819
    0.8195
    0.82
    0.8205
    0.821
    0.8215
    0.822
    0.8225
    0.823
    0.8235
    0.824
    0.8245
    0.825
    0.8255
    0.826
    0.8265
    0.827
    0.8275
    0.828
    0.8285
    0.829
    0.8295
    0.83
    0.8305
    0.831
    0.8315
    0.832
    0.8325
    0.833
    0.8335
    0.834
    0.8345
    0.835
    0.8355
    0.836
    0.8365
    0.837
    0.8375
    0.838
    0.8385
    0.839
    0.8395
    0.84
    0.8405
    0.841
    0.8415
    0.842
    0.8425
    0.843
    0.8435
    0.844
    0.8445
    0.845
    0.8455
    0.846
    0.8465
    0.847
    0.8475
    0.848
    0.8485
    0.849
    0.8495
    0.85
    0.8505
    0.851
    0.8515
    0.852
    0.8525
    0.853
    0.8535
    0.854
    0.8545
    0.855
    0.8555
    0.856
    0.8565
    0.857
    0.8575
    0.858
    0.8585
    0.859
    0.8595
    0.86
    0.8605
    0.861
    0.8615
    0.862
    0.8625
    0.863
    0.8635
    0.864
    0.8645
    0.865
    0.8655
    0.866
    0.8665
    0.867
    0.8675
    0.868
    0.8685
    0.869
    0.8695
    0.87
    0.8705
    0.871
    0.8715
    0.872
    0.8725
    0.873
    0.8735
    0.874
    0.8745
    0.875
    0.8755
    0.876
    0.8765
    0.877
    0.8775
    0.878
    0.8785
    0.879
    0.8795
    0.88
    0.8805
    0.881
    0.8815
    0.882
    0.8825
    0.883
    0.8835
    0.884
    0.8845
    0.885
    0.8855
    0.886
    0.8865
    0.887
    0.8875
    0.888
    0.8885
    0.889
    0.8895
    0.89
    0.8905
    0.891
    0.8915
    0.892
    0.8925
    0.893
    0.8935
    0.894
    0.8945
    0.895
    0.8955
    0.896
    0.8965
    0.897
    0.8975
    0.898
    0.8985
    0.899
    0.8995
    0.9
    0.9005
    0.901
    0.9015
    0.902
    0.9025
    0.903
    0.9035
    0.904
    0.9045
    0.905
    0.9055
    0.906
    0.9065
    0.907
    0.9075
    0.908
    0.9085
    0.909
    0.9095
    0.91
    0.9105
    0.911
    0.9115
    0.912
    0.9125
    0.913
    0.9135
    0.914
    0.9145
    0.915
    0.9155
    0.916
    0.9165
    0.917
    0.9175
    0.918
    0.9185
    0.919
    0.9195
    0.92
    0.9205
    0.921
    0.9215
    0.922
    0.9225
    0.923
    0.9235
    0.924
    0.9245
    0.925
    0.9255
    0.926
    0.9265
    0.927
    0.9275
    0.928
    0.9285
    0.929
    0.9295
    0.93
    0.9305
    0.931
    0.9315
    0.932
    0.9325
    0.933
    0.9335
    0.934
    0.9345
    0.935
    0.9355
    0.936
    0.9365
    0.937
    0.9375
    0.938
    0.9385
    0.939
    0.9395
    0.94
    0.9405
    0.941
    0.9415
    0.942
    0.9425
    0.943
    0.9435
    0.944
    0.9445
    0.945
    0.9455
    0.946
    0.9465
    0.947
    0.9475
    0.948
    0.9485
    0.949
    0.9495
    0.95
    0.9505
    0.951
    0.9515
    0.952
    0.9525
    0.953
    0.9535
    0.954
    0.9545
    0.955
    0.9555
    0.956
    0.9565
    0.957
    0.9575
    0.958
    0.9585
    0.959
    0.9595
    0.96
    0.9605
    0.961
    0.9615
    0.962
    0.9625
    0.963
    0.9635
    0.964
    0.9645
    0.965
    0.9655
    0.966
    0.9665
    0.967
    0.9675
    0.968
    0.9685
    0.969
    0.9695
    0.97
    0.9705
    0.971
    0.9715
    0.972
    0.9725
    0.973
    0.9735
    0.974
    0.9745
    0.975
    0.9755
    0.976
    0.9765
    0.977
    0.9775
    0.978
    0.9785
    0.979
    0.9795
    0.98
    0.9805
    0.981
    0.9815
    0.982
    0.9825
    0.983
    0.9835
    0.984
    0.9845
    0.985
    0.9855
    0.986
    0.9865
    0.987
    0.9875
    0.988
    0.9885
    0.989
    0.9895
    0.99
    0.9905
    0.991
    0.9915
    0.992
    0.9925
    0.993
    0.9935
    0.994
    0.9945
    0.995
    0.9955
    0.996
    0.9965
    0.997
    0.9975
    0.998
    0.9985
    0.999
    0.9995



```python
avg_test_mse = np.mean(test_mse_array,axis=1)
avg_train_mse = np.mean(train_mse_array,axis=1)
avg_test_accu = np.mean(test_accu_array,axis=1)
avg_train_accu = np.mean(train_accu_array,axis=1)

err_test_mse  = np.std(test_mse_array, axis=1) / np.sqrt(split_realisations)
err_train_mse  = np.std(train_mse_array, axis=1) / np.sqrt(split_realisations)
err_test_accu = np.std(test_accu_array, axis=1) / np.sqrt(split_realisations)
err_train_accu  = np.std(train_accu_array, axis=1) / np.sqrt(split_realisations)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,avg_test_accu, label="test",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_test_accu-err_test_accu, avg_test_accu+err_test_accu)

ax1.plot(n_pca_comp_ls,avg_train_accu, label="train",ls="dashed",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_train_accu-err_train_accu, avg_train_accu+err_train_accu)

ax1.set_ylim(0.73,0.75)

ax1.set_ylabel('Avg metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,avg_test_mse, label="test",c="green")
ax1.fill_between(n_pca_comp_ls, avg_test_mse-err_test_mse, avg_test_mse+err_test_mse)

ax2.plot(n_pca_comp_ls,avg_train_mse, label="train",ls="dashed",c="green")
ax1.fill_between(n_pca_comp_ls, avg_train_mse-err_train_mse, avg_train_mse+err_train_mse)

ax2.set_ylabel('Avg MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max avg test acc="+str(round(max(avg_test_accu),4))+" at n_comp="+str(n_pca_comp_ls[list(avg_test_accu).index(max(avg_test_accu))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_93_0.png)
    


Okay, let's try then `PCA_n=22` and take it from there. Here's a plot with all my submissions:


```python
test_accus = [0.7372781509514295, 0.7372266203920959, 0.7380890945496226,0.7390256249523817,0.7385244220332879,0.7377275167466955, 0.7376576670402171, 0.7379971832389312,0.7363969993867265,0.7360243522354202]
pca_com = [15,16,17,18,19,20,21, 22,24,29]
plt.plot(pca_com,test_accus)
plt.scatter(pca_com,test_accus)
plt.axhline([0.74],ls="--")
# plt.ylim(0.735,0.745)
plt.ylabel("test accurancy")
plt.xlabel("pca_com")

```




    Text(0.5, 0, 'pca_com')




    
![png](acroscarrillo_work_files/acroscarrillo_work_95_1.png)
    


might be worth trying `PCA_n < 19`, but we will leave it here. What a great challenge!



## Model analysis and interpretation

Before we put a stop to this challenge, let's take a moment to see what we can learn from our model. It is clear that in reality we hace $N$ independent linear models for each liquid return so it should sufices to study one particular liquid return, naturally: `RET_139`. Let's take a look at the $\beta$ coefficients of the model:


```python
# again we pick 139 as our example asset.
X_train_139, Y_train_139 = data_to_train(X_train,Y_train,X_supp,139)

pca = PCA(n_components=19) # best test
pca.fit( X_train_139 )
X_train_139_PCA = pca.transform(X_train_139)

# add intercept 
X_train_139_PCA = np.hstack((X_train_139_PCA, np.ones((1,X_train_139_PCA.shape[0])).T))

# no regu
betas = inv( X_train_139_PCA.T @ X_train_139_PCA ) @ X_train_139_PCA.T @ Y_train_139

m, beta = betas[0], betas[1:]

plt.plot(beta)
plt.scatter(range(len(beta)),beta)
plt.axhline(y=0,c="black")
plt.xticks(range(0,len(beta),2))
plt.ylim(-max(beta)-max(beta)/10,max(beta)+max(beta)/10)
plt.xlabel("PCA component n")
plt.xlabel("beta_n")
plt.grid(axis='x', color='0.95')
plt.title("Intercept = "+str(m[0]))

```




    Text(0.5, 1.0, 'Intercept = -0.0008091638702157965')




    
![png](acroscarrillo_work_files/acroscarrillo_work_98_1.png)
    


Interestingly, the components that contribute the most to the prediction of the model are the 9th and the 16th components and there is very little intercept. (Note how the magnitude of the $\beta$'s is already within reason: this is why adding regularisation didnt seem to help.) Let's take a look at why this might be. Well, first off, the average returns of `RET_139` are


```python
mean_ret = np.mean(Y_train_139)
print() # skip a line
mean_ret
```

    





    0.00018195198700454712



which is indeed a number of the same order of magnitude as the intercept (albeit negative), so it is reasonable to think that this is compensating for this. Next, we have that the 9th and 16th components have weights like:


```python
fig, axs = plt.subplots(1, 3, figsize=(12, 4), layout='constrained',sharex=True,sharey=True)

axs[0].plot(range(100),pca.components_[0][0:-3:4],alpha=0.5,label="CLASS_1")
axs[0].plot(range(100),pca.components_[0][1:-2:4],alpha=0.5,label="CLASS_2")
axs[0].plot(range(100),pca.components_[0][2:-1:4],alpha=0.5,label="CLASS_3")
axs[0].plot(range(100),pca.components_[0][3::4],alpha=0.5,label="CLASS_4")
axs[0].axhline(y=0,c="black")
axs[0].set_title("0th PCA eigenvector")
axs[0].set_ylabel("feature weight")
axs[0].set_xlabel("features")

axs[1].plot(range(100),pca.components_[9][0:-3:4],alpha=0.5,label="CLASS_1")
axs[1].plot(range(100),pca.components_[9][1:-2:4],alpha=0.5,label="CLASS_2")
axs[1].plot(range(100),pca.components_[9][2:-1:4],alpha=0.5,label="CLASS_3")
axs[1].plot(range(100),pca.components_[9][3::4],alpha=0.5,label="CLASS_4")
axs[1].axhline(y=0,c="black")
axs[1].set_title("9th PCA eigenvector")
axs[1].set_ylabel("feature weight")
axs[1].set_xlabel("features")

axs[2].plot(range(100),pca.components_[16][0:-3:4],alpha=0.5,label="CLASS_1")
axs[2].plot(range(100),pca.components_[16][1:-2:4],alpha=0.5,label="CLASS_2")
axs[2].plot(range(100),pca.components_[16][2:-1:4],alpha=0.5,label="CLASS_3")
axs[2].plot(range(100),pca.components_[16][3::4],alpha=0.5,label="CLASS_4")
axs[2].axhline(y=0,c="black")
axs[2].set_title("16th PCA eigenvector")
axs[2].set_ylabel("feature weight")
axs[2].set_xlabel("features")

plt.legend(loc="upper right")

```




    <matplotlib.legend.Legend at 0x2915113d0>




    
![png](acroscarrillo_work_files/acroscarrillo_work_102_1.png)
    


But this is not very illuminating since we have little intuition behind these objects. Instead, lets plot at what the weights of the total transformation is doing, PCA + linear model included:


```python
weights_139 = pca.components_.T @ beta + m


plt.plot(range(100),weights_139[0:-3:4],alpha=0.5,label="CLASS_1")
plt.plot(range(100),weights_139[1:-2:4],alpha=0.5,label="CLASS_2")
plt.plot(range(100),weights_139[2:-1:4],alpha=0.5,label="CLASS_3")
plt.plot(range(100),weights_139[3::4],alpha=0.5,label="CLASS_4")
plt.ylim(-max(weights_139)-max(weights_139)/10, max(weights_139)+ max(weights_139)/10)
plt.axhline(y=0,c="black")
plt.title("Linear model transformation")
plt.ylabel("feature weight")
plt.xlabel("RET_n")

plt.legend(loc="upper right")

```




    <matplotlib.legend.Legend at 0x2c500ba90>




    
![png](acroscarrillo_work_files/acroscarrillo_work_104_1.png)
    


Okay, this is much more interpretable! Again, seems like returns weighted with `CLASS_2` returns tend to have a larger predictive power for this particular return which as explained at the beggining this is likely to do with the fact that its the class the closest with the average. From this data we can extract which are the top 10 returns that contribute the most to our target returns.


```python
calss_1_f = list(np.flip(np.argsort(class_2_w,axis=0))[0:10][:,0])
calss_1_f
```




    [0, 50, 77, 78, 21, 12, 37, 19, 46, 20]




```python
fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
class_1_w = weights_139[0:-3:4]
class_2_w = weights_139[1:-2:4]
class_3_w = weights_139[2:-1:4]
class_4_w = weights_139[3::4]

calss_1_f = list(np.flip(np.argsort(class_1_w,axis=0))[0:10][:,0])
calss_1_f = [str(item) for item in calss_1_f]
class_1_w_sorted = np.abs(np.sort(class_1_w,axis=0)[0:10][:,0])
axs[0,0].set_title("Class 1 top contributing RETs")
axs[0,0].bar(calss_1_f,class_1_w_sorted)
axs[0,0].set_xlabel("RET_n")
axs[0,0].set_ylabel("abs(Weight)")

calss_2_f = list(np.flip(np.argsort(class_2_w,axis=0))[0:10][:,0])
calss_2_f = [str(item) for item in calss_2_f]
class_2_w_sorted = np.abs(np.sort(class_2_w,axis=0)[0:10][:,0])
axs[0,1].set_title("Class 2 top contributing RETs")
axs[0,1].bar(calss_2_f,class_2_w_sorted)
axs[0,1].set_xlabel("RET_n")
axs[0,1].set_ylabel("abs(Weight)")

calss_3_f = list(np.flip(np.argsort(class_3_w,axis=0))[0:10][:,0])
calss_3_f = [str(item) for item in calss_3_f]
class_3_w_sorted = np.abs(np.sort(class_3_w,axis=0)[0:10][:,0])
axs[1,0].set_title("Class 3 top contributing RETs")
axs[1,0].bar(calss_3_f,class_3_w_sorted)
axs[1,0].set_xlabel("RET_n")
axs[1,0].set_ylabel("abs(Weight)")

calss_4_f = list(np.flip(np.argsort(class_4_w,axis=0))[0:10][:,0])
calss_4_f = [str(item) for item in calss_4_f]
class_4_w_sorted = np.abs(np.sort(class_4_w,axis=0)[0:10][:,0])
axs[1,1].set_title("Class 4 top contributing RETs")
axs[1,1].bar(calss_3_f,class_3_w_sorted)
axs[1,1].set_xlabel("RET_n")
axs[1,1].set_ylabel("abs(Weight)")
```




    Text(0, 0.5, 'abs(Weight)')




    
![png](acroscarrillo_work_files/acroscarrillo_work_107_1.png)
    


So `RET_0`, `RET_50` and `RET_77` are the most significant. 

We could be a bit more careful and adjust this by the mean of each return but the idea is the same so we will leave this short analysis here. 

## Addendum: pre-standarsing the returns
It is worth noting that standarising the returns before weighting them does not improve the model accurancy: in fact one could argue it makes it worse by making it more noisy. This is the reason we have decided not to pre-standarise the return data.


```python
percent_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
n_pca_comp_ls = list(range(1,150,1))
mse_tot_test_ls_ls = []
mse_tot_train_ls_ls = []
test_accu_ls_ls = []
train_accu_ls_ls = []
for percent in percent_ls:
    test_perc = percent
    mse_tot_test_ls = []
    mse_tot_train_ls = []
    test_accu_ls = []
    train_accu_ls = []
    for n_pca_comp in n_pca_comp_ls:
        test_accu, train_accu, mse_tot_test, mse_tot_train = fast_linear_model_stats(model_PCAs,df_x_train,df_y_train,test_perc,n_pca_comp)
        test_accu_ls.append(test_accu)
        train_accu_ls.append(train_accu)
        mse_tot_test_ls.append(mse_tot_test)
        mse_tot_train_ls.append(mse_tot_train)
        # print(n_pca_comp/250, test_accu)

    test_accu_ls_ls.append(test_accu_ls)
    train_accu_ls_ls.append(train_accu_ls)
    mse_tot_test_ls_ls.append(mse_tot_test_ls)
    mse_tot_train_ls_ls.append(mse_tot_train_ls)
    print(percent/len(percent_ls))


fig, axs = plt.subplots(3, 2, figsize=(12, 14), layout='constrained',sharex=True,sharey=True)
i=0
for ax in axs.flat:
    test_perc = percent_ls[i]
    mse_tot_test_ls = mse_tot_test_ls_ls[i]
    mse_tot_train_ls = mse_tot_train_ls_ls[i]
    test_accu_ls = test_accu_ls_ls[i]
    train_accu_ls = train_accu_ls_ls[i]
    i += 1

    ax_twin = ax.twinx()

    ax.plot(n_pca_comp_ls,test_accu_ls, label="test",c="blue")
    ax.plot(n_pca_comp_ls,train_accu_ls, label="train",ls="dashed",c="blue")
    ax.set_ylabel('Metric accurancy',c="blue")

    ax_twin.plot(n_pca_comp_ls,mse_tot_test_ls, label="test",c="green")
    ax_twin.plot(n_pca_comp_ls,mse_tot_train_ls, label="train",ls="dashed",c="green")
    ax_twin.set_ylabel('MSE',c="green")

    ax.set_xlabel("no. PCA components")
    plt.title("test split="+str(test_perc)+". Max test acc="+str(round(max(test_accu_ls),4))+" at n_comp="+str(n_pca_comp_ls[test_accu_ls.index(max(test_accu_ls))]))
    plt.legend()

plt.show()
```

    0.008333333333333333
    0.016666666666666666
    0.024999999999999998
    0.03333333333333333
    0.041666666666666664
    0.049999999999999996



    
![png](acroscarrillo_work_files/acroscarrillo_work_111_1.png)
    



```python
test_perc = 0.1

split_realisations = 50
n_pca_comp_ls = list(range(15,70,2))

test_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
train_mse_array = np.zeros((len(n_pca_comp_ls), split_realisations))
test_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))
train_accu_array = np.zeros((len(n_pca_comp_ls), split_realisations))

for j in range(split_realisations):
    for (k,n_pca_comp) in enumerate(n_pca_comp_ls):
        test_accu, train_accu, mse_tot_test, mse_tot_train = fast_linear_model_stats(model_PCAs,df_x_train,df_y_train,test_perc,n_pca_comp)
        test_mse_array[k,j] = mse_tot_test
        train_mse_array[k,j] = mse_tot_train
        test_accu_array[k,j] = test_accu
        train_accu_array[k,j] = train_accu

    print(j/split_realisations)


# DELETE

avg_test_mse = np.mean(test_mse_array,axis=1)
avg_train_mse = np.mean(train_mse_array,axis=1)
avg_test_accu = np.mean(test_accu_array,axis=1)
avg_train_accu = np.mean(train_accu_array,axis=1)

err_test_mse  = np.std(test_mse_array, axis=1) / np.sqrt(split_realisations)
err_train_mse  = np.std(train_mse_array, axis=1) / np.sqrt(split_realisations)
err_test_accu = np.std(test_accu_array, axis=1) / np.sqrt(split_realisations)
err_train_accu  = np.std(train_accu_array, axis=1) / np.sqrt(split_realisations)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,avg_test_accu, label="test",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_test_accu-err_test_accu, avg_test_accu+err_test_accu)

ax1.plot(n_pca_comp_ls,avg_train_accu, label="train",ls="dashed",c="blue")
ax1.fill_between(n_pca_comp_ls, avg_train_accu-err_train_accu, avg_train_accu+err_train_accu)

ax1.set_ylim(0.73,0.75)

ax1.set_ylabel('Avg metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,avg_test_mse, label="test",c="green")
ax1.fill_between(n_pca_comp_ls, avg_test_mse-err_test_mse, avg_test_mse+err_test_mse)

ax2.plot(n_pca_comp_ls,avg_train_mse, label="train",ls="dashed",c="green")
ax1.fill_between(n_pca_comp_ls, avg_train_mse-err_train_mse, avg_train_mse+err_train_mse)

ax2.set_ylabel('Avg MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max avg test acc="+str(round(max(avg_test_accu),4))+" at n_comp="+str(n_pca_comp_ls[list(avg_test_accu).index(max(avg_test_accu))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_112_0.png)
    



```python
# again we pick 139 as our example asset.
X_train_139, Y_train_139 = data_to_train(X_train,Y_train,X_supp,139)

pca = PCA(n_components=19) # best test
pca.fit( X_train_139 )
X_train_139_PCA = pca.transform(X_train_139)

# add intercept 
X_train_139_PCA = np.hstack((X_train_139_PCA, np.ones((1,X_train_139_PCA.shape[0])).T))

# no regu
betas = inv( X_train_139_PCA.T @ X_train_139_PCA ) @ X_train_139_PCA.T @ Y_train_139

m, beta = betas[0], betas[1:]

plt.plot(beta)
plt.scatter(range(len(beta)),beta)
plt.axhline(y=0,c="black")
plt.xticks(range(0,len(beta),2))
plt.ylim(-max(beta)-max(beta)/10,max(beta)+max(beta)/10)
plt.xlabel("PCA component n")
plt.xlabel("beta_n")
plt.grid(axis='x', color='0.95')
plt.title("Intercept = "+str(m[0]))

```




    Text(0.5, 1.0, 'Intercept = -1.6852234148115668e-05')




    
![png](acroscarrillo_work_files/acroscarrillo_work_113_1.png)
    



```python
weights_139 = pca.components_.T @ beta + m


plt.plot(range(100),weights_139[0:-3:4],alpha=0.5,label="CLASS_1")
plt.plot(range(100),weights_139[1:-2:4],alpha=0.5,label="CLASS_2")
plt.plot(range(100),weights_139[2:-1:4],alpha=0.5,label="CLASS_3")
plt.plot(range(100),weights_139[3::4],alpha=0.5,label="CLASS_4")
plt.ylim(-max(weights_139)-max(weights_139)/10, max(weights_139)+ max(weights_139)/10)
plt.axhline(y=0,c="black")
plt.title("Linear model transformation")
plt.ylabel("feature weight")
plt.xlabel("RET_n")

plt.legend(loc="upper right")

```




    <matplotlib.legend.Legend at 0x297bbba90>




    
![png](acroscarrillo_work_files/acroscarrillo_work_114_1.png)
    


---
---
---
---

# Further models and ideas tested (inferior performance) 

Here's a rather undocumented set of further ideas tested. From simply adding regularisation to the above model all the way to deep learning models passing by classification algorithms like random forests. Perhaps the only place I saw potential was with random forests (treating the problem as a $+1$, $-1$ classification problem) and deep learning if a genetic algorithm can be made to work (the given metric is non differentiable so this slightly dificulted my attemps and could be overcomed by a genetic algorith which does not rely on gradients).


```python
percent_ls = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

n_pca_comp_ls = list(range(1,150,1))
# mse_tot_test_ls_ls = []
# mse_tot_train_ls_ls = []
# test_accu_ls_ls = []
# train_accu_ls_ls = []
for percent in percent_ls:
    test_perc = percent
    mse_tot_test_ls = []
    mse_tot_train_ls = []
    test_accu_ls = []
    train_accu_ls = []
    for n_pca_comp in n_pca_comp_ls:
        test_accu, train_accu, mse_tot_test, mse_tot_train = linear_model_stats(model_PCAs,X_train,Y_train,X_supp,test_perc,n_pca_comp)
        test_accu_ls.append(test_accu)
        train_accu_ls.append(train_accu)
        mse_tot_test_ls.append(mse_tot_test)
        mse_tot_train_ls.append(mse_tot_train)
        # print(n_pca_comp/250, test_accu)

    test_accu_ls_ls.append(test_accu_ls)
    train_accu_ls_ls.append(train_accu_ls)
    mse_tot_test_ls_ls.append(mse_tot_test_ls)
    mse_tot_train_ls_ls.append(mse_tot_train_ls)
    print(percent/len(percent_ls))
```


```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(n_pca_comp_ls,test_accu_ls, label="test",c="blue")
ax1.plot(n_pca_comp_ls,train_accu_ls, label="train",ls="dashed",c="blue")
ax1.set_ylabel('Metric accurancy',c="blue")

ax2.plot(n_pca_comp_ls,mse_tot_test_ls, label="test",c="green")
ax2.plot(n_pca_comp_ls,mse_tot_train_ls, label="train",ls="dashed",c="green")
ax2.set_ylabel('MSE',c="green")

ax1.set_xlabel("no. PCA components")
plt.title("test split="+str(test_perc)+". Max test acc="+str(round(max(test_accu_ls),4))+" at n_comp="+str(n_pca_comp_ls[test_accu_ls.index(max(test_accu_ls))]))
plt.legend()
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_117_0.png)
    


Look at that phase transition like change in the data at around $100$ components! This makes sense if we remember the discontinuous jump in the spectrum at around that very same point. Beautiful!

Let's be a bit more thorough by looking at what happens as we adjust the test/train split:


```python
percent_ls = list(i/100 for i in range(10,50,5))

n_pca_comp_ls = list(range(1,250,5))
# mse_tot_test_ls_ls = []
# mse_tot_train_ls_ls = []
# test_accu_ls_ls = []
# train_accu_ls_ls = []

for percent in percent_ls:
    test_perc = percent
    mse_tot_test_ls = []
    mse_tot_train_ls = []
    test_accu_ls = []
    train_accu_ls = []
    for n_pca_comp in n_pca_comp_ls:
        test_accu, train_accu, mse_tot_test, mse_tot_train = linear_model_stats(X_train,Y_train,X_supp,test_percent=test_perc,PCA_percent=n_pca_comp)
        test_accu_ls.append(test_accu)
        train_accu_ls.append(train_accu)
        mse_tot_test_ls.append(mse_tot_test)
        mse_tot_train_ls.append(mse_tot_train)
        # print(n_pca_comp/250, test_accu)

    test_accu_ls_ls.append(test_accu_ls)
    train_accu_ls_ls.append(train_accu_ls)
    mse_tot_test_ls_ls.append(mse_tot_test_ls)
    mse_tot_train_ls_ls.append(mse_tot_train_ls)
    print(percent/.45)
```

    0.22222222222222224
    0.3333333333333333
    0.4444444444444445
    0.5555555555555556
    0.6666666666666666
    0.7777777777777777
    0.888888888888889
    1.0



```python

fig, axs = plt.subplots(4, 2, figsize=(12, 14), layout='constrained',sharex=True,sharey=True)
i=0
for ax in axs.flat:
    test_perc = percent_ls[i]
    mse_tot_test_ls = mse_tot_test_ls_ls[i]
    mse_tot_train_ls = mse_tot_train_ls_ls[i]
    test_accu_ls = test_accu_ls_ls[i]
    train_accu_ls = train_accu_ls_ls[i]
    i += 1

    ax_twin = ax.twinx()

    ax.plot(n_pca_comp_ls,test_accu_ls, label="test",c="blue")
    ax.plot(n_pca_comp_ls,train_accu_ls, label="train",ls="dashed",c="blue")
    ax.set_ylabel('Metric accurancy',c="blue")

    ax_twin.plot(n_pca_comp_ls,mse_tot_test_ls, label="test",c="green")
    ax_twin.plot(n_pca_comp_ls,mse_tot_train_ls, label="train",ls="dashed",c="green")
    ax_twin.set_ylabel('MSE',c="green")

    ax.set_xlabel("no. PCA components")
    plt.title("test split="+str(test_perc)+". Max test acc="+str(round(max(test_accu_ls),4))+" at n_comp="+str(n_pca_comp_ls[test_accu_ls.index(max(test_accu_ls))]))
    plt.legend()

plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_120_0.png)
    


So sure, a larger test set makes the data less noisy but it also brings down the test accurancy a bit over a percent. Having said that, it doesnt look like there is any notable qualitative changes across these different test splits. 

All in all, it seems like this model, as it stands, is limited to a $ <73\%$ accurancy when using $\sim 100$ principal components. We could try to push this model a bit further (by perhaps rethinking our feature engenieering or with a more clever loss function) however we cannot resist to see what happens with a deep learning model. With this feature engenieering we did, the data set is no longer as small as we had previously thought since now the data is not simply repeated data due to our weighting process. In the next model we will see we can go the extra mile with deep neural networks and squeeze an extra $1\%$ or $2\%$.

# As a classification

### Random forests



```python
def data_to_train_pd(X_train,Y_train,supp_df,liquid_ID):
    train_df = weight_returns(X_train,supp_df,liquid_ID)
    ID_list = train_df["ID"]
    test_df = Y_train.loc[Y_train['ID'].isin(ID_list)]    
    
    train_array = train_df.drop(columns=["ID"])
    test_array = test_df.drop(columns=["ID"])
    return train_array, test_array

df_x_train = pd.DataFrame()
df_y_train = pd.DataFrame()
for target_ID in X_train["ID_TARGET"].unique():
     df_temp_x, df_temp_y = data_to_train_pd(X_train,Y_train,X_supp,target_ID)
     df_x_train = pd.concat([df_x_train,df_temp_x])
     df_y_train = pd.concat([df_y_train,df_temp_y])
     
df_x_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RET_216_0</th>
      <th>RET_216_1</th>
      <th>RET_216_2</th>
      <th>RET_216_3</th>
      <th>RET_238_0</th>
      <th>RET_238_1</th>
      <th>RET_238_2</th>
      <th>RET_238_3</th>
      <th>RET_45_0</th>
      <th>RET_45_1</th>
      <th>...</th>
      <th>RET_95_2</th>
      <th>RET_95_3</th>
      <th>RET_162_0</th>
      <th>RET_162_1</th>
      <th>RET_162_2</th>
      <th>RET_162_3</th>
      <th>RET_297_0</th>
      <th>RET_297_1</th>
      <th>RET_297_2</th>
      <th>RET_297_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002870</td>
      <td>0.003778</td>
      <td>0.002758</td>
      <td>0.002506</td>
      <td>0.006588</td>
      <td>0.008672</td>
      <td>0.006331</td>
      <td>0.005526</td>
      <td>0.002608</td>
      <td>0.003530</td>
      <td>...</td>
      <td>0.000133</td>
      <td>0.000129</td>
      <td>-0.001189</td>
      <td>-0.001181</td>
      <td>-0.000824</td>
      <td>-0.000705</td>
      <td>-0.020538</td>
      <td>-0.022636</td>
      <td>-0.018484</td>
      <td>-0.016595</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.018436</td>
      <td>0.024268</td>
      <td>0.017716</td>
      <td>0.016094</td>
      <td>-0.001504</td>
      <td>-0.001980</td>
      <td>-0.001446</td>
      <td>-0.001262</td>
      <td>-0.011448</td>
      <td>-0.015497</td>
      <td>...</td>
      <td>-0.001091</td>
      <td>-0.001055</td>
      <td>0.002120</td>
      <td>0.002104</td>
      <td>0.001468</td>
      <td>0.001257</td>
      <td>0.001417</td>
      <td>0.001562</td>
      <td>0.001275</td>
      <td>0.001145</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-0.008749</td>
      <td>-0.011517</td>
      <td>-0.008407</td>
      <td>-0.007638</td>
      <td>0.005322</td>
      <td>0.007005</td>
      <td>0.005114</td>
      <td>0.004464</td>
      <td>0.026942</td>
      <td>0.036471</td>
      <td>...</td>
      <td>0.002529</td>
      <td>0.002445</td>
      <td>0.003585</td>
      <td>0.003558</td>
      <td>0.002483</td>
      <td>0.002125</td>
      <td>0.007266</td>
      <td>0.008008</td>
      <td>0.006539</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>300</th>
      <td>-0.023964</td>
      <td>-0.031545</td>
      <td>-0.023028</td>
      <td>-0.020920</td>
      <td>-0.002458</td>
      <td>-0.003235</td>
      <td>-0.002362</td>
      <td>-0.002062</td>
      <td>-0.004778</td>
      <td>-0.006468</td>
      <td>...</td>
      <td>-0.002648</td>
      <td>-0.002560</td>
      <td>0.001672</td>
      <td>0.001660</td>
      <td>0.001158</td>
      <td>0.000991</td>
      <td>0.006732</td>
      <td>0.007419</td>
      <td>0.006058</td>
      <td>0.005439</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.020806</td>
      <td>0.027387</td>
      <td>0.019992</td>
      <td>0.018163</td>
      <td>-0.060123</td>
      <td>-0.079141</td>
      <td>-0.057773</td>
      <td>-0.050430</td>
      <td>0.000171</td>
      <td>0.000231</td>
      <td>...</td>
      <td>-0.002687</td>
      <td>-0.002598</td>
      <td>-0.000618</td>
      <td>-0.000614</td>
      <td>-0.000428</td>
      <td>-0.000366</td>
      <td>-0.012285</td>
      <td>-0.013541</td>
      <td>-0.011057</td>
      <td>-0.009927</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266703</th>
      <td>-0.012034</td>
      <td>-0.012034</td>
      <td>-0.010992</td>
      <td>-0.011348</td>
      <td>-0.021459</td>
      <td>-0.021459</td>
      <td>-0.019601</td>
      <td>-0.019828</td>
      <td>-0.004046</td>
      <td>-0.003884</td>
      <td>...</td>
      <td>0.000539</td>
      <td>0.000581</td>
      <td>-0.003203</td>
      <td>-0.002199</td>
      <td>-0.001680</td>
      <td>-0.001651</td>
      <td>-0.002273</td>
      <td>-0.001773</td>
      <td>-0.001574</td>
      <td>-0.001608</td>
    </tr>
    <tr>
      <th>266802</th>
      <td>0.001643</td>
      <td>0.001643</td>
      <td>0.001501</td>
      <td>0.001549</td>
      <td>0.018795</td>
      <td>0.018795</td>
      <td>0.017167</td>
      <td>0.017367</td>
      <td>0.047766</td>
      <td>0.045851</td>
      <td>...</td>
      <td>-0.001797</td>
      <td>-0.001937</td>
      <td>0.031023</td>
      <td>0.021304</td>
      <td>0.016272</td>
      <td>0.015997</td>
      <td>0.001066</td>
      <td>0.000831</td>
      <td>0.000738</td>
      <td>0.000754</td>
    </tr>
    <tr>
      <th>266901</th>
      <td>0.010200</td>
      <td>0.010200</td>
      <td>0.009317</td>
      <td>0.009619</td>
      <td>-0.003231</td>
      <td>-0.003231</td>
      <td>-0.002952</td>
      <td>-0.002986</td>
      <td>-0.016627</td>
      <td>-0.015961</td>
      <td>...</td>
      <td>0.001269</td>
      <td>0.001368</td>
      <td>-0.012262</td>
      <td>-0.008420</td>
      <td>-0.006431</td>
      <td>-0.006322</td>
      <td>-0.004896</td>
      <td>-0.003819</td>
      <td>-0.003391</td>
      <td>-0.003464</td>
    </tr>
    <tr>
      <th>267000</th>
      <td>0.000532</td>
      <td>0.000532</td>
      <td>0.000486</td>
      <td>0.000501</td>
      <td>-0.006588</td>
      <td>-0.006588</td>
      <td>-0.006017</td>
      <td>-0.006087</td>
      <td>-0.048504</td>
      <td>-0.046559</td>
      <td>...</td>
      <td>-0.002448</td>
      <td>-0.002639</td>
      <td>-0.008945</td>
      <td>-0.006143</td>
      <td>-0.004692</td>
      <td>-0.004612</td>
      <td>-0.000498</td>
      <td>-0.000388</td>
      <td>-0.000345</td>
      <td>-0.000352</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>0.025293</td>
      <td>0.025293</td>
      <td>0.023102</td>
      <td>0.023851</td>
      <td>-0.003277</td>
      <td>-0.003277</td>
      <td>-0.002993</td>
      <td>-0.003028</td>
      <td>-0.026191</td>
      <td>-0.025140</td>
      <td>...</td>
      <td>-0.000421</td>
      <td>-0.000453</td>
      <td>-0.015923</td>
      <td>-0.010935</td>
      <td>-0.008352</td>
      <td>-0.008211</td>
      <td>0.005732</td>
      <td>0.004471</td>
      <td>0.003970</td>
      <td>0.004056</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 400 columns</p>
</div>




```python
from sklearn import ensemble 

def class_accu(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    sign_sum = 0.5 * np.abs( np.sign(y_true) + np.sign(y_pred) )
    sum_array = np.multiply( np.abs(y_true), sign_sum ) 
    sum_term = np.sum( sum_array )
    norm_term = np.sum( np.abs(y_true) )
    return np.divide( sum_term,norm_term ) 


train_x, test_x, train_y, test_y = train_test_split(df_x_train,df_y_train, test_size=0.1)
y_train_signed, y_test_signed = train_y, test_y

y_train_signed["RET_TARGET"], y_test_signed["RET_TARGET"] = np.sign(y_train_signed["RET_TARGET"]), np.sign(y_test_signed["RET_TARGET"])

# dtr = ensemble.RandomForestRegressor(n_estimators=10,min_samples_leaf=25,max_depth = 20,verbose=2)
# regr = dtr.fit(train_x, y_train_signed)
pred_y =  np.reshape(  regr.predict(test_x), (len(test_y),1))

acc = class_accu(test_y,pred_y)
acc
```

Okay, looks promising, but can we do better than with our linear regression ? Well, let's see if by tunning the depth of our trees we can fine tune further.


```python
train_x, test_x, train_y, test_y = train_test_split(df_x_train,df_y_train, test_size=0.1)
y_train_signed, y_test_signed = train_y, test_y

y_train_signed["RET_TARGET"], y_test_signed["RET_TARGET"] = np.sign(y_train_signed["RET_TARGET"]), np.sign(y_test_signed["RET_TARGET"])

max_depth_ls = list(range(5,100,5))
acc_ls = []
for m in max_depth_ls:
    dtr = ensemble.RandomForestRegressor(n_estimators=10,max_depth=m)
    regr = dtr.fit(train_x, train_y)
    pred_y = regr.predict(test_x)

    pred_y =  np.reshape(  regr.predict(test_x), (len(test_y),1))
    acc = class_accu(test_y,pred_y) 

    acc_ls.append(acc)
    print(m/max(max_depth_ls), acc)
```

    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.05263157894736842 0.6175964058405091


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.10526315789473684 0.6392362411081992


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.15789473684210525 0.6452639460876076


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.21052631578947367 0.6420816173717708


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.2631578947368421 0.6344814676151255


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.3157894736842105 0.6244664919505803


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.3684210526315789 0.6228378884312992


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.42105263157894735 0.6213777611381505


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.47368421052631576 0.6231374017222014


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.5263157894736842 0.6247660052414826


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.5789473684210527 0.6268438786971172


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.631578947368421 0.6272182703107451


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.6842105263157895 0.6264882066641707


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.7368421052631579 0.6241108199176338


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.7894736842105263 0.6233620366903782


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.8421052631578947 0.6232309996256084


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.8947368421052632 0.6242044178210409


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    0.9473684210526315 0.6238300262074129


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/base.py:1351: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return fit_method(estimator, *args, **kwargs)


    1.0 0.6254399101460127



```python
from sklearn.linear_model import LogisticRegression

def class_accu(y_true,y_pred):
    sign_sum = 0.5 * np.abs( np.sign(y_true) + np.sign(y_pred) )
    sum_array = np.multiply( np.abs(y_true), sign_sum ) 
    sum_term = np.sum( sum_array )
    norm_term = np.sum( np.abs(y_true) )
    return np.divide( sum_term,norm_term ) 

train_x, test_x, train_y, test_y = train_test_split(df_x_train,df_y_train, test_size=0.1)
y_train_signed, y_test_signed = train_y, test_y

y_train_signed["RET_TARGET"], y_test_signed["RET_TARGET"] = np.sign(y_train_signed["RET_TARGET"]), np.sign(y_test_signed["RET_TARGET"])

clf = LogisticRegression(penalty="l1",solver="liblinear").fit(train_x, y_train_signed)

pred_y = np.reshape( clf.predict(test_x), (len(pred_y),1))

acc = class_accu(test_y,pred_y)
acc
```

    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/sklearn/utils/validation.py:1229: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/numpy/core/fromnumeric.py:86: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
      return reduction(axis=axis, out=out, **passkwargs)





    RET_TARGET    0.622613
    dtype: float64




```python
binary_cross = tf.keras.losses.BinaryCrossentropy()

def custom_binary_cross(y_true,y_pred):
    y_true_bin = ( tf.math.sign(y_true) + 1 )/2
    return tf.math.multiply(binary_cross(y_true_bin,y_pred),tf.abs(y_true))

y_true_test = tf.constant([1.0, -3.5, 4.1, 0.1])
y_pred_test = tf.constant([0.5, -2, -3, 0.01])
custom_binary_cross(y_true_test,y_pred_test)
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([ 5.1808143, 18.13285  , 21.241339 ,  0.5180814], dtype=float32)>




```python
def sign_cnt(x,k=1e3):
    return 2*(tf.sigmoid(x*k)-0.5)

def abs_cnt(x,k=1e3):
    return x*tf.sigmoid(x*k)-x*tf.sigmoid(-x*k) 

def custom_acc(y_true,y_pred):
    sign_sum = 0.5 * tf.abs( tf.math.sign(y_true) + tf.math.sign(y_pred) )
    sum_array = tf.math.multiply( tf.abs(y_true), sign_sum ) 
    sum_term = tf.reduce_sum( sum_array )
    norm_term = tf.reduce_sum( tf.abs(y_true) )
    return tf.math.divide( sum_term,norm_term )

def class_accu(y_true,y_pred):
    sign_sum = 0.5 * np.abs( np.sign(y_true) + np.sign(y_pred) )
    sum_array = np.multiply( np.abs(y_true), sign_sum ) 
    sum_term = np.sum( sum_array )
    norm_term = np.sum( np.abs(y_true) )
    return np.divide( sum_term,norm_term ) 

def custom_loss(y_true,y_pred):
    sign_sum = 0.5 * abs_cnt( sign_cnt(y_true) + sign_cnt(y_pred) )
    sum_array = tf.math.multiply( tf.abs(y_true), sign_sum ) 
    sum_term = tf.reduce_sum( sum_array )
    norm_term = tf.reduce_sum( tf.abs(y_true) )
    return -tf.math.divide( sum_term,norm_term ) 

binary_cross = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def custom_binary_cross(y_true,y_pred):
    y_true_bin = ( tf.math.sign(y_true) + 1 )/2
    return tf.math.multiply(binary_cross(y_true_bin,y_pred),tf.abs(y_true))


train_x, test_x, train_y, test_y = train_test_split(df_x_train,df_y_train, test_size=0.1)

train_y_signed = train_y.copy()
train_y_signed["RET_TARGET"] = (np.sign(train_y["RET_TARGET"]) + 1)/2
test_y_signed = test_y.copy()
test_y_signed["RET_TARGET"] = (np.sign(test_y["RET_TARGET"]) + 1)/2

model = keras.Sequential([
    layers.Dense(400, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(100, activation='relu'), 
    # layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(1,activation="sigmoid"),
])

L_rate = 1e-4

binary_accu = tf.keras.metrics.BinaryAccuracy()

model.compile(loss=custom_binary_cross,
            optimizer=tf.keras.optimizers.legacy.Adam(L_rate), metrics=[binary_accu])


checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="binary_accuracy", 
    # monitor="custom_acc", # save best custom_acc
    # monitor="mse", # save best custom_acc
    mode='max', # the larger custom_acc the better
    save_weights_only=True,
    save_best_only=True
    )


## if comented is because I wanted to delete the verbose output of the fitter.
history = model.fit(train_x, train_y_signed, validation_data=(test_x,test_y_signed), epochs=200,verbose=2,batch_size=100, callbacks=[model_checkpoint_callback])

# model.predict(test_x)
# test_y
```

    Epoch 1/200
    2404/2404 - 4s - loss: 0.2675 - binary_accuracy: 0.4757 - val_loss: 0.2661 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 2/200
    2404/2404 - 4s - loss: 0.2651 - binary_accuracy: 0.4756 - val_loss: 0.2659 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 3/200
    2404/2404 - 4s - loss: 0.2648 - binary_accuracy: 0.4756 - val_loss: 0.2656 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 4/200
    2404/2404 - 4s - loss: 0.2644 - binary_accuracy: 0.4756 - val_loss: 0.2654 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 5/200
    2404/2404 - 4s - loss: 0.2641 - binary_accuracy: 0.4756 - val_loss: 0.2652 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 6/200
    2404/2404 - 4s - loss: 0.2638 - binary_accuracy: 0.4756 - val_loss: 0.2651 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 7/200
    2404/2404 - 4s - loss: 0.2636 - binary_accuracy: 0.4756 - val_loss: 0.2649 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 8/200
    2404/2404 - 4s - loss: 0.2634 - binary_accuracy: 0.4757 - val_loss: 0.2648 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 9/200
    2404/2404 - 4s - loss: 0.2631 - binary_accuracy: 0.4757 - val_loss: 0.2648 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 10/200
    2404/2404 - 4s - loss: 0.2630 - binary_accuracy: 0.4757 - val_loss: 0.2647 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 11/200
    2404/2404 - 4s - loss: 0.2628 - binary_accuracy: 0.4757 - val_loss: 0.2646 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 12/200
    2404/2404 - 4s - loss: 0.2628 - binary_accuracy: 0.4758 - val_loss: 0.2646 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 13/200
    2404/2404 - 4s - loss: 0.2625 - binary_accuracy: 0.4759 - val_loss: 0.2646 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 14/200
    2404/2404 - 4s - loss: 0.2625 - binary_accuracy: 0.4758 - val_loss: 0.2645 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 15/200
    2404/2404 - 4s - loss: 0.2624 - binary_accuracy: 0.4759 - val_loss: 0.2645 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 16/200
    2404/2404 - 4s - loss: 0.2623 - binary_accuracy: 0.4760 - val_loss: 0.2644 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 17/200
    2404/2404 - 4s - loss: 0.2622 - binary_accuracy: 0.4759 - val_loss: 0.2643 - val_binary_accuracy: 0.4798 - 4s/epoch - 2ms/step
    Epoch 18/200
    2404/2404 - 4s - loss: 0.2621 - binary_accuracy: 0.4759 - val_loss: 0.2644 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 19/200
    2404/2404 - 4s - loss: 0.2620 - binary_accuracy: 0.4760 - val_loss: 0.2643 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 20/200
    2404/2404 - 4s - loss: 0.2619 - binary_accuracy: 0.4761 - val_loss: 0.2644 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 21/200
    2404/2404 - 4s - loss: 0.2618 - binary_accuracy: 0.4761 - val_loss: 0.2644 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 22/200
    2404/2404 - 4s - loss: 0.2617 - binary_accuracy: 0.4761 - val_loss: 0.2643 - val_binary_accuracy: 0.4797 - 4s/epoch - 2ms/step
    Epoch 23/200
    2404/2404 - 4s - loss: 0.2616 - binary_accuracy: 0.4761 - val_loss: 0.2643 - val_binary_accuracy: 0.4800 - 4s/epoch - 2ms/step
    Epoch 24/200
    2404/2404 - 4s - loss: 0.2616 - binary_accuracy: 0.4762 - val_loss: 0.2641 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 25/200
    2404/2404 - 4s - loss: 0.2616 - binary_accuracy: 0.4761 - val_loss: 0.2642 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 26/200
    2404/2404 - 4s - loss: 0.2615 - binary_accuracy: 0.4761 - val_loss: 0.2642 - val_binary_accuracy: 0.4797 - 4s/epoch - 2ms/step
    Epoch 27/200
    2404/2404 - 4s - loss: 0.2615 - binary_accuracy: 0.4761 - val_loss: 0.2641 - val_binary_accuracy: 0.4798 - 4s/epoch - 2ms/step
    Epoch 28/200
    2404/2404 - 4s - loss: 0.2614 - binary_accuracy: 0.4763 - val_loss: 0.2641 - val_binary_accuracy: 0.4797 - 4s/epoch - 2ms/step
    Epoch 29/200
    2404/2404 - 4s - loss: 0.2613 - binary_accuracy: 0.4762 - val_loss: 0.2643 - val_binary_accuracy: 0.4795 - 4s/epoch - 2ms/step
    Epoch 30/200
    2404/2404 - 4s - loss: 0.2612 - binary_accuracy: 0.4762 - val_loss: 0.2642 - val_binary_accuracy: 0.4798 - 4s/epoch - 2ms/step
    Epoch 31/200
    2404/2404 - 4s - loss: 0.2612 - binary_accuracy: 0.4763 - val_loss: 0.2645 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 32/200
    2404/2404 - 4s - loss: 0.2612 - binary_accuracy: 0.4763 - val_loss: 0.2642 - val_binary_accuracy: 0.4796 - 4s/epoch - 2ms/step
    Epoch 33/200
    2404/2404 - 4s - loss: 0.2611 - binary_accuracy: 0.4763 - val_loss: 0.2642 - val_binary_accuracy: 0.4800 - 4s/epoch - 2ms/step
    Epoch 34/200



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[124], line 71
         59 model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
         60     filepath=checkpoint_filepath,
         61     monitor="binary_accuracy", 
       (...)
         66     save_best_only=True
         67     )
         70 ## if comented is because I wanted to delete the verbose output of the fitter.
    ---> 71 history = model.fit(train_x, train_y_signed, validation_data=(test_x,test_y_signed), epochs=200,verbose=2,batch_size=100, callbacks=[model_checkpoint_callback])
         73 # model.predict(test_x)
         74 # test_y


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py:65, in filter_traceback.<locals>.error_handler(*args, **kwargs)
         63 filtered_tb = None
         64 try:
    ---> 65     return fn(*args, **kwargs)
         66 except Exception as e:
         67     filtered_tb = _process_traceback_frames(e.__traceback__)


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/keras/src/engine/training.py:1807, in Model.fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)
       1799 with tf.profiler.experimental.Trace(
       1800     "train",
       1801     epoch_num=epoch,
       (...)
       1804     _r=1,
       1805 ):
       1806     callbacks.on_train_batch_begin(step)
    -> 1807     tmp_logs = self.train_function(iterator)
       1808     if data_handler.should_sync:
       1809         context.async_wait()


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/util/traceback_utils.py:150, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        148 filtered_tb = None
        149 try:
    --> 150   return fn(*args, **kwargs)
        151 except Exception as e:
        152   filtered_tb = _process_traceback_frames(e.__traceback__)


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:832, in Function.__call__(self, *args, **kwds)
        829 compiler = "xla" if self._jit_compile else "nonXla"
        831 with OptionalXlaContext(self._jit_compile):
    --> 832   result = self._call(*args, **kwds)
        834 new_tracing_count = self.experimental_get_tracing_count()
        835 without_tracing = (tracing_count == new_tracing_count)


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:868, in Function._call(self, *args, **kwds)
        865   self._lock.release()
        866   # In this case we have created variables on the first call, so we run the
        867   # defunned version which is guaranteed to never create variables.
    --> 868   return tracing_compilation.call_function(
        869       args, kwds, self._no_variable_creation_config
        870   )
        871 elif self._variable_creation_config is not None:
        872   # Release the lock early so that multiple threads can perform the call
        873   # in parallel.
        874   self._lock.release()


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:139, in call_function(args, kwargs, tracing_options)
        137 bound_args = function.function_type.bind(*args, **kwargs)
        138 flat_inputs = function.function_type.unpack_inputs(bound_args)
    --> 139 return function._call_flat(  # pylint: disable=protected-access
        140     flat_inputs, captured_inputs=function.captured_inputs
        141 )


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1323, in ConcreteFunction._call_flat(self, tensor_inputs, captured_inputs)
       1319 possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
       1320 if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
       1321     and executing_eagerly):
       1322   # No tape is watching; skip to running the function.
    -> 1323   return self._inference_function.call_preflattened(args)
       1324 forward_backward = self._select_forward_and_backward_functions(
       1325     args,
       1326     possible_gradient_type,
       1327     executing_eagerly)
       1328 forward_function, args_with_tangents = forward_backward.forward()


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:216, in AtomicFunction.call_preflattened(self, args)
        214 def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
        215   """Calls with flattened tensor inputs and returns the structured output."""
    --> 216   flat_outputs = self.call_flat(*args)
        217   return self.function_type.pack_output(flat_outputs)


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:251, in AtomicFunction.call_flat(self, *args)
        249 with record.stop_recording():
        250   if self._bound_context.executing_eagerly():
    --> 251     outputs = self._bound_context.call_function(
        252         self.name,
        253         list(args),
        254         len(self.function_type.flat_outputs),
        255     )
        256   else:
        257     outputs = make_call_op_in_graph(
        258         self,
        259         list(args),
        260         self._bound_context.function_call_options.as_attrs(),
        261     )


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/context.py:1486, in Context.call_function(self, name, tensor_inputs, num_outputs)
       1484 cancellation_context = cancellation.context()
       1485 if cancellation_context is None:
    -> 1486   outputs = execute.execute(
       1487       name.decode("utf-8"),
       1488       num_outputs=num_outputs,
       1489       inputs=tensor_inputs,
       1490       attrs=attrs,
       1491       ctx=self,
       1492   )
       1493 else:
       1494   outputs = execute.execute_with_cancellation(
       1495       name.decode("utf-8"),
       1496       num_outputs=num_outputs,
       (...)
       1500       cancellation_manager=cancellation_context,
       1501   )


    File ~/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/tensorflow/python/eager/execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         51 try:
         52   ctx.ensure_initialized()
    ---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         54                                       inputs, attrs, num_outputs)
         55 except core._NotOkStatusException as e:
         56   if name is not None:


    KeyboardInterrupt: 



```python
model.load_weights(checkpoint_filepath)
y_pred = np.sign(model.predict(test_x) - 0.5)
print(y_pred,test_y)
class_accu(test_y,y_pred)
```

    835/835 [==============================] - 0s 557us/step
    [[1.]
     [1.]
     [1.]
     ...
     [1.]
     [1.]
     [1.]]         RET_TARGET
    46543    -0.028975
    174467    0.021741
    96265    -0.000382
    104283   -0.005367
    175816   -0.039988
    ...            ...
    163075   -0.004709
    241806    0.003969
    27659    -0.010495
    132936   -0.006145
    10702    -0.006305
    
    [26710 rows x 1 columns]


    /Users/alex/Documents/Projects/QRT_challlenge_2/QRT_2/lib/python3.11/site-packages/numpy/core/fromnumeric.py:86: FutureWarning: The behavior of DataFrame.sum with axis=None is deprecated, in a future version this will reduce over both axes and return a scalar. To retain the old behavior, pass axis=0 (or do not pass axis)
      return reduction(axis=axis, out=out, **passkwargs)





    RET_TARGET    0.516637
    dtype: float64




```python
test_y
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46543</th>
      <td>-0.028975</td>
    </tr>
    <tr>
      <th>174467</th>
      <td>0.021741</td>
    </tr>
    <tr>
      <th>96265</th>
      <td>-0.000382</td>
    </tr>
    <tr>
      <th>104283</th>
      <td>-0.005367</td>
    </tr>
    <tr>
      <th>175816</th>
      <td>-0.039988</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>163075</th>
      <td>-0.004709</td>
    </tr>
    <tr>
      <th>241806</th>
      <td>0.003969</td>
    </tr>
    <tr>
      <th>27659</th>
      <td>-0.010495</td>
    </tr>
    <tr>
      <th>132936</th>
      <td>-0.006145</td>
    </tr>
    <tr>
      <th>10702</th>
      <td>-0.006305</td>
    </tr>
  </tbody>
</table>
<p>26710 rows × 1 columns</p>
</div>




```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(history.history["binary_accuracy"],label="train",ls="dashed",c="blue")
ax1.plot(history.history['val_binary_accuracy'],label="test",c="blue")
ax1.set_ylabel('binary_accuracy',c="blue")

ax2.plot(history.history["val_loss"], label="test",c="green")
ax2.plot(history.history["loss"], label="train",ls="dashed",c="green")
ax2.set_ylabel('Loss = - Metric acc',c="green")

ax1.set_title("L_rate = "+str(L_rate)+" no PCA. Max test acc = "+str(max(history.history['val_binary_accuracy'])))

ax1.axhline(0.74,c="black",ls="dashdot",alpha=0.5)
ax1.axhline(0.745,c="black",ls="dashdot",alpha=0.5)
ax1.axhline(0.75,c="black",ls="dashdot",alpha=0.5)
ax1.axvline(history.history["val_binary_accuracy"].index(max(history.history["val_binary_accuracy"])),c="black",ls="dashdot",alpha=0.5)

# ax1.set_yscale("log")
# axs[i].set_yscale("log")

ax1.set_xlabel('epoch')
plt.legend( loc='center right')
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_132_0.png)
    


## Deep learning model
We will test a particular deep learning model with and without PCA. We first define some helper functions to apply all the feature engenieering we did to the entire data set:


```python
def data_to_train_pd(X_train,Y_train,supp_df,liquid_ID):
    train_df = weight_returns(X_train,supp_df,liquid_ID)
    ID_list = train_df["ID"]
    test_df = Y_train.loc[Y_train['ID'].isin(ID_list)]    
    
    train_array = train_df.drop(columns=["ID"])
    test_array = test_df.drop(columns=["ID"])
    return train_array, test_array

df_x_train = pd.DataFrame()
df_y_train = pd.DataFrame()
for target_ID in X_train["ID_TARGET"].unique():
     df_temp_x, df_temp_y = data_to_train_pd(X_train,Y_train,X_supp,target_ID)
     df_x_train = pd.concat([df_x_train,df_temp_x])
     df_y_train = pd.concat([df_y_train,df_temp_y])
     
df_x_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RET_216_0</th>
      <th>RET_216_1</th>
      <th>RET_216_2</th>
      <th>RET_216_3</th>
      <th>RET_238_0</th>
      <th>RET_238_1</th>
      <th>RET_238_2</th>
      <th>RET_238_3</th>
      <th>RET_45_0</th>
      <th>RET_45_1</th>
      <th>...</th>
      <th>RET_95_2</th>
      <th>RET_95_3</th>
      <th>RET_162_0</th>
      <th>RET_162_1</th>
      <th>RET_162_2</th>
      <th>RET_162_3</th>
      <th>RET_297_0</th>
      <th>RET_297_1</th>
      <th>RET_297_2</th>
      <th>RET_297_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.002870</td>
      <td>0.003778</td>
      <td>0.002758</td>
      <td>0.002506</td>
      <td>0.006588</td>
      <td>0.008672</td>
      <td>0.006331</td>
      <td>0.005526</td>
      <td>0.002608</td>
      <td>0.003530</td>
      <td>...</td>
      <td>0.000133</td>
      <td>0.000129</td>
      <td>-0.001189</td>
      <td>-0.001181</td>
      <td>-0.000824</td>
      <td>-0.000705</td>
      <td>-0.020538</td>
      <td>-0.022636</td>
      <td>-0.018484</td>
      <td>-0.016595</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.018436</td>
      <td>0.024268</td>
      <td>0.017716</td>
      <td>0.016094</td>
      <td>-0.001504</td>
      <td>-0.001980</td>
      <td>-0.001446</td>
      <td>-0.001262</td>
      <td>-0.011448</td>
      <td>-0.015497</td>
      <td>...</td>
      <td>-0.001091</td>
      <td>-0.001055</td>
      <td>0.002120</td>
      <td>0.002104</td>
      <td>0.001468</td>
      <td>0.001257</td>
      <td>0.001417</td>
      <td>0.001562</td>
      <td>0.001275</td>
      <td>0.001145</td>
    </tr>
    <tr>
      <th>200</th>
      <td>-0.008749</td>
      <td>-0.011517</td>
      <td>-0.008407</td>
      <td>-0.007638</td>
      <td>0.005322</td>
      <td>0.007005</td>
      <td>0.005114</td>
      <td>0.004464</td>
      <td>0.026942</td>
      <td>0.036471</td>
      <td>...</td>
      <td>0.002529</td>
      <td>0.002445</td>
      <td>0.003585</td>
      <td>0.003558</td>
      <td>0.002483</td>
      <td>0.002125</td>
      <td>0.007266</td>
      <td>0.008008</td>
      <td>0.006539</td>
      <td>0.005871</td>
    </tr>
    <tr>
      <th>300</th>
      <td>-0.023964</td>
      <td>-0.031545</td>
      <td>-0.023028</td>
      <td>-0.020920</td>
      <td>-0.002458</td>
      <td>-0.003235</td>
      <td>-0.002362</td>
      <td>-0.002062</td>
      <td>-0.004778</td>
      <td>-0.006468</td>
      <td>...</td>
      <td>-0.002648</td>
      <td>-0.002560</td>
      <td>0.001672</td>
      <td>0.001660</td>
      <td>0.001158</td>
      <td>0.000991</td>
      <td>0.006732</td>
      <td>0.007419</td>
      <td>0.006058</td>
      <td>0.005439</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.020806</td>
      <td>0.027387</td>
      <td>0.019992</td>
      <td>0.018163</td>
      <td>-0.060123</td>
      <td>-0.079141</td>
      <td>-0.057773</td>
      <td>-0.050430</td>
      <td>0.000171</td>
      <td>0.000231</td>
      <td>...</td>
      <td>-0.002687</td>
      <td>-0.002598</td>
      <td>-0.000618</td>
      <td>-0.000614</td>
      <td>-0.000428</td>
      <td>-0.000366</td>
      <td>-0.012285</td>
      <td>-0.013541</td>
      <td>-0.011057</td>
      <td>-0.009927</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>266703</th>
      <td>-0.012034</td>
      <td>-0.012034</td>
      <td>-0.010992</td>
      <td>-0.011348</td>
      <td>-0.021459</td>
      <td>-0.021459</td>
      <td>-0.019601</td>
      <td>-0.019828</td>
      <td>-0.004046</td>
      <td>-0.003884</td>
      <td>...</td>
      <td>0.000539</td>
      <td>0.000581</td>
      <td>-0.003203</td>
      <td>-0.002199</td>
      <td>-0.001680</td>
      <td>-0.001651</td>
      <td>-0.002273</td>
      <td>-0.001773</td>
      <td>-0.001574</td>
      <td>-0.001608</td>
    </tr>
    <tr>
      <th>266802</th>
      <td>0.001643</td>
      <td>0.001643</td>
      <td>0.001501</td>
      <td>0.001549</td>
      <td>0.018795</td>
      <td>0.018795</td>
      <td>0.017167</td>
      <td>0.017367</td>
      <td>0.047766</td>
      <td>0.045851</td>
      <td>...</td>
      <td>-0.001797</td>
      <td>-0.001937</td>
      <td>0.031023</td>
      <td>0.021304</td>
      <td>0.016272</td>
      <td>0.015997</td>
      <td>0.001066</td>
      <td>0.000831</td>
      <td>0.000738</td>
      <td>0.000754</td>
    </tr>
    <tr>
      <th>266901</th>
      <td>0.010200</td>
      <td>0.010200</td>
      <td>0.009317</td>
      <td>0.009619</td>
      <td>-0.003231</td>
      <td>-0.003231</td>
      <td>-0.002952</td>
      <td>-0.002986</td>
      <td>-0.016627</td>
      <td>-0.015961</td>
      <td>...</td>
      <td>0.001269</td>
      <td>0.001368</td>
      <td>-0.012262</td>
      <td>-0.008420</td>
      <td>-0.006431</td>
      <td>-0.006322</td>
      <td>-0.004896</td>
      <td>-0.003819</td>
      <td>-0.003391</td>
      <td>-0.003464</td>
    </tr>
    <tr>
      <th>267000</th>
      <td>0.000532</td>
      <td>0.000532</td>
      <td>0.000486</td>
      <td>0.000501</td>
      <td>-0.006588</td>
      <td>-0.006588</td>
      <td>-0.006017</td>
      <td>-0.006087</td>
      <td>-0.048504</td>
      <td>-0.046559</td>
      <td>...</td>
      <td>-0.002448</td>
      <td>-0.002639</td>
      <td>-0.008945</td>
      <td>-0.006143</td>
      <td>-0.004692</td>
      <td>-0.004612</td>
      <td>-0.000498</td>
      <td>-0.000388</td>
      <td>-0.000345</td>
      <td>-0.000352</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>0.025293</td>
      <td>0.025293</td>
      <td>0.023102</td>
      <td>0.023851</td>
      <td>-0.003277</td>
      <td>-0.003277</td>
      <td>-0.002993</td>
      <td>-0.003028</td>
      <td>-0.026191</td>
      <td>-0.025140</td>
      <td>...</td>
      <td>-0.000421</td>
      <td>-0.000453</td>
      <td>-0.015923</td>
      <td>-0.010935</td>
      <td>-0.008352</td>
      <td>-0.008211</td>
      <td>0.005732</td>
      <td>0.004471</td>
      <td>0.003970</td>
      <td>0.004056</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 400 columns</p>
</div>




```python
df_y_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.022351</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.008354</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.012218</td>
    </tr>
    <tr>
      <th>300</th>
      <td>-0.004456</td>
    </tr>
    <tr>
      <th>400</th>
      <td>0.008788</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>266703</th>
      <td>0.041931</td>
    </tr>
    <tr>
      <th>266802</th>
      <td>-0.019130</td>
    </tr>
    <tr>
      <th>266901</th>
      <td>0.015095</td>
    </tr>
    <tr>
      <th>267000</th>
      <td>-0.009511</td>
    </tr>
    <tr>
      <th>267099</th>
      <td>0.005056</td>
    </tr>
  </tbody>
</table>
<p>267100 rows × 1 columns</p>
</div>



With that out of the way, we will now "analytically continue" the custom metric on which the challenge is assed to maximise our chances of performing well. This is necesary as the given metric is not differentiable ($\textrm{sign}(\cdot)$ and $|\cdot|$ functions are not differentiable). The idea is very simple:
$$
\textrm{sign}(x) \xrightarrow{A.C} 2\sigma(kx)-1 \quad \text{ and } \quad |x| \xrightarrow{A.C}  x\sigma(kx)-x\sigma(-kx),
$$
where $\sigma(kx)$ is the sigmoid function with $k$ playing the role of a smoothing parameter (the larger it is the sharper the fit). We code those two up and we create a `custom_loss` function by simply minimising the negative of the given metric $-f(y^t,y^p)$.


```python
def sign_cnt(x,k=1e3):
    return 2*(tf.sigmoid(x*k)-0.5)

def abs_cnt(x,k=1e3):
    return x*tf.sigmoid(x*k)-x*tf.sigmoid(-x*k) 

def custom_acc(y_true,y_pred):
    sign_sum = 0.5 * tf.abs( tf.math.sign(y_true) + tf.math.sign(y_pred) )
    sum_array = tf.math.multiply( tf.abs(y_true), sign_sum ) 
    sum_term = tf.reduce_sum( sum_array )
    norm_term = tf.reduce_sum( tf.abs(y_true) )
    return tf.math.divide( sum_term,norm_term )

def custom_loss(y_true,y_pred):
    sign_sum = 0.5 * abs_cnt( sign_cnt(y_true) + sign_cnt(y_pred) )
    sum_array = tf.math.multiply( tf.abs(y_true), sign_sum ) 
    sum_term = tf.reduce_sum( sum_array )
    norm_term = tf.reduce_sum( tf.abs(y_true) )
    return -tf.math.divide( sum_term,norm_term ) 

y_true_test = tf.constant([1.0, -3.5, 4.1, 0.1])
y_pred_test = tf.constant([0.5, -2, -3, 0.01])
print( custom_acc(y_true_test,y_pred_test) )
print( custom_loss(y_true_test,y_pred_test) )
```

    tf.Tensor(0.52873564, shape=(), dtype=float32)
    tf.Tensor(-0.52873516, shape=(), dtype=float32)


Which as you can see in the two print statements above, it very nicely approximates the exact value for large $k$. For the actual deep learning model we follow a standard setup with some hidden layers, `relu` as activation function and some dropout to prevent overfitting. 


```python
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
    tf.keras.layers.InputLayer(400),
    # layers.Dense(400, activation='linear'),
    # layers.Dense(100, activation='relu'),
    layers.Dense(500, activation='relu'),
    layers.Dense(300, activation='relu'),
    layers.Dropout(.1),
    layers.Dense(1,activation="linear") # linear is the default
])

L_rate = 1e-5
# model.compile(loss="mse",
#             optimizer=tf.keras.optimizers.legacy.Adam(L_rate), metrics=["mse",custom_acc])

model.compile(loss=custom_loss,
            optimizer=tf.keras.optimizers.legacy.Adam(L_rate), metrics=[custom_acc])

checkpoint_filepath = '/tmp/ckpt/checkpoint.model.keras'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="custom_acc", # save best custom_acc
    # monitor="mse", # save best custom_acc
    mode='max', # the larger custom_acc the better
    save_weights_only=True,
    save_best_only=True
    )


# pca = PCA(n_components=200) # explain % of variability
# pca.fit( df_x_train )
# df_x_train_pca = pca.transform(df_x_train)

idx_shuffled = np.random.permutation(df_x_train.index)
df_x_train = df_x_train.reindex(idx_shuffled)
df_y_train = df_y_train.reindex(idx_shuffled)

## if comented is because I wanted to delete the verbose output of the fitter.
history = model.fit(df_x_train, df_y_train, validation_split = 0.1, epochs=100,verbose=2,batch_size=1000, callbacks=[model_checkpoint_callback])
```

    Epoch 1/100
    241/241 - 2s - loss: -6.5959e-01 - custom_acc: 0.6612 - val_loss: -6.8553e-01 - val_custom_acc: 0.6864 - 2s/epoch - 8ms/step
    Epoch 2/100
    241/241 - 2s - loss: -6.8779e-01 - custom_acc: 0.6890 - val_loss: -6.9072e-01 - val_custom_acc: 0.6929 - 2s/epoch - 7ms/step
    Epoch 3/100
    241/241 - 2s - loss: -6.9246e-01 - custom_acc: 0.6931 - val_loss: -6.9249e-01 - val_custom_acc: 0.6944 - 2s/epoch - 7ms/step
    Epoch 4/100
    241/241 - 2s - loss: -6.9498e-01 - custom_acc: 0.6959 - val_loss: -6.9267e-01 - val_custom_acc: 0.6937 - 2s/epoch - 6ms/step
    Epoch 5/100
    241/241 - 2s - loss: -6.9738e-01 - custom_acc: 0.6986 - val_loss: -6.9361e-01 - val_custom_acc: 0.6948 - 2s/epoch - 6ms/step
    Epoch 6/100
    241/241 - 2s - loss: -6.9846e-01 - custom_acc: 0.6994 - val_loss: -6.9531e-01 - val_custom_acc: 0.6967 - 2s/epoch - 6ms/step
    Epoch 7/100
    241/241 - 2s - loss: -6.9987e-01 - custom_acc: 0.7008 - val_loss: -6.9603e-01 - val_custom_acc: 0.6976 - 2s/epoch - 7ms/step
    Epoch 8/100
    241/241 - 2s - loss: -7.0048e-01 - custom_acc: 0.7013 - val_loss: -6.9757e-01 - val_custom_acc: 0.6993 - 2s/epoch - 6ms/step
    Epoch 9/100
    241/241 - 2s - loss: -7.0152e-01 - custom_acc: 0.7025 - val_loss: -6.9763e-01 - val_custom_acc: 0.6993 - 2s/epoch - 7ms/step
    Epoch 10/100
    241/241 - 2s - loss: -7.0211e-01 - custom_acc: 0.7027 - val_loss: -6.9788e-01 - val_custom_acc: 0.6994 - 2s/epoch - 6ms/step
    Epoch 11/100
    241/241 - 2s - loss: -7.0281e-01 - custom_acc: 0.7036 - val_loss: -6.9796e-01 - val_custom_acc: 0.6992 - 2s/epoch - 7ms/step
    Epoch 12/100
    241/241 - 2s - loss: -7.0373e-01 - custom_acc: 0.7049 - val_loss: -6.9834e-01 - val_custom_acc: 0.6997 - 2s/epoch - 6ms/step
    Epoch 13/100
    241/241 - 2s - loss: -7.0484e-01 - custom_acc: 0.7060 - val_loss: -6.9857e-01 - val_custom_acc: 0.6993 - 2s/epoch - 6ms/step
    Epoch 14/100
    241/241 - 2s - loss: -7.0489e-01 - custom_acc: 0.7058 - val_loss: -6.9898e-01 - val_custom_acc: 0.7004 - 2s/epoch - 7ms/step
    Epoch 15/100
    241/241 - 2s - loss: -7.0566e-01 - custom_acc: 0.7065 - val_loss: -6.9961e-01 - val_custom_acc: 0.7011 - 2s/epoch - 6ms/step
    Epoch 16/100
    241/241 - 2s - loss: -7.0604e-01 - custom_acc: 0.7069 - val_loss: -6.9887e-01 - val_custom_acc: 0.6997 - 2s/epoch - 6ms/step
    Epoch 17/100
    241/241 - 2s - loss: -7.0622e-01 - custom_acc: 0.7073 - val_loss: -6.9908e-01 - val_custom_acc: 0.7003 - 2s/epoch - 6ms/step
    Epoch 18/100
    241/241 - 2s - loss: -7.0689e-01 - custom_acc: 0.7078 - val_loss: -6.9922e-01 - val_custom_acc: 0.7004 - 2s/epoch - 6ms/step
    Epoch 19/100
    241/241 - 2s - loss: -7.0757e-01 - custom_acc: 0.7089 - val_loss: -6.9901e-01 - val_custom_acc: 0.6999 - 2s/epoch - 6ms/step
    Epoch 20/100
    241/241 - 2s - loss: -7.0770e-01 - custom_acc: 0.7089 - val_loss: -6.9937e-01 - val_custom_acc: 0.7002 - 2s/epoch - 6ms/step
    Epoch 21/100
    241/241 - 2s - loss: -7.0813e-01 - custom_acc: 0.7090 - val_loss: -6.9978e-01 - val_custom_acc: 0.7009 - 2s/epoch - 6ms/step
    Epoch 22/100
    241/241 - 1s - loss: -7.0850e-01 - custom_acc: 0.7097 - val_loss: -7.0013e-01 - val_custom_acc: 0.7012 - 1s/epoch - 6ms/step
    Epoch 23/100
    241/241 - 2s - loss: -7.0928e-01 - custom_acc: 0.7103 - val_loss: -7.0043e-01 - val_custom_acc: 0.7013 - 2s/epoch - 7ms/step
    Epoch 24/100
    241/241 - 2s - loss: -7.0954e-01 - custom_acc: 0.7107 - val_loss: -7.0041e-01 - val_custom_acc: 0.7008 - 2s/epoch - 6ms/step
    Epoch 25/100
    241/241 - 2s - loss: -7.1003e-01 - custom_acc: 0.7109 - val_loss: -7.0082e-01 - val_custom_acc: 0.7012 - 2s/epoch - 6ms/step
    Epoch 26/100
    241/241 - 2s - loss: -7.1002e-01 - custom_acc: 0.7112 - val_loss: -7.0075e-01 - val_custom_acc: 0.7013 - 2s/epoch - 7ms/step
    Epoch 27/100
    241/241 - 2s - loss: -7.1055e-01 - custom_acc: 0.7119 - val_loss: -7.0140e-01 - val_custom_acc: 0.7021 - 2s/epoch - 6ms/step
    Epoch 28/100
    241/241 - 2s - loss: -7.1049e-01 - custom_acc: 0.7116 - val_loss: -7.0113e-01 - val_custom_acc: 0.7026 - 2s/epoch - 6ms/step
    Epoch 29/100
    241/241 - 2s - loss: -7.1082e-01 - custom_acc: 0.7116 - val_loss: -7.0241e-01 - val_custom_acc: 0.7042 - 2s/epoch - 7ms/step
    Epoch 30/100
    241/241 - 2s - loss: -7.1108e-01 - custom_acc: 0.7124 - val_loss: -7.0269e-01 - val_custom_acc: 0.7039 - 2s/epoch - 7ms/step
    Epoch 31/100
    241/241 - 2s - loss: -7.1188e-01 - custom_acc: 0.7134 - val_loss: -7.0356e-01 - val_custom_acc: 0.7051 - 2s/epoch - 7ms/step
    Epoch 32/100
    241/241 - 2s - loss: -7.1251e-01 - custom_acc: 0.7136 - val_loss: -7.0359e-01 - val_custom_acc: 0.7053 - 2s/epoch - 6ms/step
    Epoch 33/100
    241/241 - 2s - loss: -7.1270e-01 - custom_acc: 0.7138 - val_loss: -7.0378e-01 - val_custom_acc: 0.7055 - 2s/epoch - 7ms/step
    Epoch 34/100
    241/241 - 2s - loss: -7.1290e-01 - custom_acc: 0.7142 - val_loss: -7.0381e-01 - val_custom_acc: 0.7051 - 2s/epoch - 6ms/step
    Epoch 35/100
    241/241 - 2s - loss: -7.1312e-01 - custom_acc: 0.7146 - val_loss: -7.0373e-01 - val_custom_acc: 0.7054 - 2s/epoch - 6ms/step
    Epoch 36/100
    241/241 - 2s - loss: -7.1311e-01 - custom_acc: 0.7139 - val_loss: -7.0376e-01 - val_custom_acc: 0.7051 - 2s/epoch - 6ms/step
    Epoch 37/100
    241/241 - 2s - loss: -7.1329e-01 - custom_acc: 0.7147 - val_loss: -7.0327e-01 - val_custom_acc: 0.7049 - 2s/epoch - 6ms/step
    Epoch 38/100
    241/241 - 2s - loss: -7.1386e-01 - custom_acc: 0.7151 - val_loss: -7.0377e-01 - val_custom_acc: 0.7052 - 2s/epoch - 6ms/step
    Epoch 39/100
    241/241 - 2s - loss: -7.1389e-01 - custom_acc: 0.7152 - val_loss: -7.0395e-01 - val_custom_acc: 0.7051 - 2s/epoch - 6ms/step
    Epoch 40/100
    241/241 - 2s - loss: -7.1381e-01 - custom_acc: 0.7151 - val_loss: -7.0401e-01 - val_custom_acc: 0.7056 - 2s/epoch - 6ms/step
    Epoch 41/100
    241/241 - 2s - loss: -7.1440e-01 - custom_acc: 0.7158 - val_loss: -7.0425e-01 - val_custom_acc: 0.7063 - 2s/epoch - 7ms/step
    Epoch 42/100
    241/241 - 2s - loss: -7.1456e-01 - custom_acc: 0.7159 - val_loss: -7.0399e-01 - val_custom_acc: 0.7058 - 2s/epoch - 6ms/step
    Epoch 43/100
    241/241 - 2s - loss: -7.1476e-01 - custom_acc: 0.7157 - val_loss: -7.0433e-01 - val_custom_acc: 0.7056 - 2s/epoch - 6ms/step
    Epoch 44/100
    241/241 - 2s - loss: -7.1514e-01 - custom_acc: 0.7165 - val_loss: -7.0471e-01 - val_custom_acc: 0.7063 - 2s/epoch - 7ms/step
    Epoch 45/100
    241/241 - 2s - loss: -7.1485e-01 - custom_acc: 0.7162 - val_loss: -7.0504e-01 - val_custom_acc: 0.7068 - 2s/epoch - 6ms/step
    Epoch 46/100
    241/241 - 2s - loss: -7.1502e-01 - custom_acc: 0.7163 - val_loss: -7.0479e-01 - val_custom_acc: 0.7055 - 2s/epoch - 6ms/step
    Epoch 47/100
    241/241 - 2s - loss: -7.1556e-01 - custom_acc: 0.7167 - val_loss: -7.0464e-01 - val_custom_acc: 0.7060 - 2s/epoch - 6ms/step
    Epoch 48/100
    241/241 - 2s - loss: -7.1553e-01 - custom_acc: 0.7168 - val_loss: -7.0466e-01 - val_custom_acc: 0.7058 - 2s/epoch - 6ms/step
    Epoch 49/100
    241/241 - 2s - loss: -7.1610e-01 - custom_acc: 0.7172 - val_loss: -7.0486e-01 - val_custom_acc: 0.7065 - 2s/epoch - 6ms/step
    Epoch 50/100
    241/241 - 2s - loss: -7.1646e-01 - custom_acc: 0.7180 - val_loss: -7.0432e-01 - val_custom_acc: 0.7053 - 2s/epoch - 6ms/step
    Epoch 51/100
    241/241 - 2s - loss: -7.1661e-01 - custom_acc: 0.7181 - val_loss: -7.0429e-01 - val_custom_acc: 0.7050 - 2s/epoch - 6ms/step
    Epoch 52/100
    241/241 - 2s - loss: -7.1649e-01 - custom_acc: 0.7179 - val_loss: -7.0449e-01 - val_custom_acc: 0.7059 - 2s/epoch - 6ms/step
    Epoch 53/100
    241/241 - 2s - loss: -7.1699e-01 - custom_acc: 0.7182 - val_loss: -7.0502e-01 - val_custom_acc: 0.7071 - 2s/epoch - 6ms/step
    Epoch 54/100
    241/241 - 2s - loss: -7.1692e-01 - custom_acc: 0.7182 - val_loss: -7.0567e-01 - val_custom_acc: 0.7077 - 2s/epoch - 7ms/step
    Epoch 55/100
    241/241 - 2s - loss: -7.1712e-01 - custom_acc: 0.7183 - val_loss: -7.0554e-01 - val_custom_acc: 0.7079 - 2s/epoch - 7ms/step
    Epoch 56/100
    241/241 - 2s - loss: -7.1713e-01 - custom_acc: 0.7182 - val_loss: -7.0556e-01 - val_custom_acc: 0.7076 - 2s/epoch - 6ms/step
    Epoch 57/100
    241/241 - 2s - loss: -7.1769e-01 - custom_acc: 0.7191 - val_loss: -7.0531e-01 - val_custom_acc: 0.7073 - 2s/epoch - 7ms/step
    Epoch 58/100
    241/241 - 2s - loss: -7.1768e-01 - custom_acc: 0.7190 - val_loss: -7.0542e-01 - val_custom_acc: 0.7068 - 2s/epoch - 7ms/step
    Epoch 59/100
    241/241 - 2s - loss: -7.1777e-01 - custom_acc: 0.7190 - val_loss: -7.0570e-01 - val_custom_acc: 0.7081 - 2s/epoch - 6ms/step
    Epoch 60/100
    241/241 - 2s - loss: -7.1773e-01 - custom_acc: 0.7192 - val_loss: -7.0529e-01 - val_custom_acc: 0.7071 - 2s/epoch - 7ms/step
    Epoch 61/100
    241/241 - 2s - loss: -7.1790e-01 - custom_acc: 0.7190 - val_loss: -7.0491e-01 - val_custom_acc: 0.7062 - 2s/epoch - 7ms/step
    Epoch 62/100
    241/241 - 2s - loss: -7.1816e-01 - custom_acc: 0.7195 - val_loss: -7.0545e-01 - val_custom_acc: 0.7069 - 2s/epoch - 7ms/step
    Epoch 63/100
    241/241 - 2s - loss: -7.1810e-01 - custom_acc: 0.7194 - val_loss: -7.0582e-01 - val_custom_acc: 0.7078 - 2s/epoch - 7ms/step
    Epoch 64/100
    241/241 - 2s - loss: -7.1855e-01 - custom_acc: 0.7200 - val_loss: -7.0560e-01 - val_custom_acc: 0.7071 - 2s/epoch - 6ms/step
    Epoch 65/100
    241/241 - 2s - loss: -7.1842e-01 - custom_acc: 0.7198 - val_loss: -7.0575e-01 - val_custom_acc: 0.7072 - 2s/epoch - 6ms/step
    Epoch 66/100
    241/241 - 2s - loss: -7.1867e-01 - custom_acc: 0.7199 - val_loss: -7.0542e-01 - val_custom_acc: 0.7069 - 2s/epoch - 6ms/step
    Epoch 67/100
    241/241 - 2s - loss: -7.1895e-01 - custom_acc: 0.7205 - val_loss: -7.0559e-01 - val_custom_acc: 0.7071 - 2s/epoch - 6ms/step
    Epoch 68/100
    241/241 - 2s - loss: -7.1899e-01 - custom_acc: 0.7202 - val_loss: -7.0579e-01 - val_custom_acc: 0.7076 - 2s/epoch - 6ms/step
    Epoch 69/100
    241/241 - 2s - loss: -7.1874e-01 - custom_acc: 0.7199 - val_loss: -7.0529e-01 - val_custom_acc: 0.7072 - 2s/epoch - 6ms/step
    Epoch 70/100
    241/241 - 2s - loss: -7.1986e-01 - custom_acc: 0.7214 - val_loss: -7.0523e-01 - val_custom_acc: 0.7070 - 2s/epoch - 7ms/step
    Epoch 71/100
    241/241 - 2s - loss: -7.1912e-01 - custom_acc: 0.7203 - val_loss: -7.0570e-01 - val_custom_acc: 0.7067 - 2s/epoch - 6ms/step
    Epoch 72/100
    241/241 - 1s - loss: -7.1913e-01 - custom_acc: 0.7204 - val_loss: -7.0565e-01 - val_custom_acc: 0.7072 - 1s/epoch - 6ms/step
    Epoch 73/100
    241/241 - 2s - loss: -7.1946e-01 - custom_acc: 0.7208 - val_loss: -7.0535e-01 - val_custom_acc: 0.7072 - 2s/epoch - 6ms/step
    Epoch 74/100
    241/241 - 2s - loss: -7.1959e-01 - custom_acc: 0.7208 - val_loss: -7.0598e-01 - val_custom_acc: 0.7075 - 2s/epoch - 6ms/step
    Epoch 75/100
    241/241 - 2s - loss: -7.1969e-01 - custom_acc: 0.7208 - val_loss: -7.0564e-01 - val_custom_acc: 0.7071 - 2s/epoch - 6ms/step
    Epoch 76/100
    241/241 - 2s - loss: -7.1998e-01 - custom_acc: 0.7214 - val_loss: -7.0564e-01 - val_custom_acc: 0.7064 - 2s/epoch - 6ms/step
    Epoch 77/100
    241/241 - 2s - loss: -7.2046e-01 - custom_acc: 0.7220 - val_loss: -7.0568e-01 - val_custom_acc: 0.7071 - 2s/epoch - 6ms/step
    Epoch 78/100
    241/241 - 2s - loss: -7.2025e-01 - custom_acc: 0.7215 - val_loss: -7.0601e-01 - val_custom_acc: 0.7072 - 2s/epoch - 6ms/step
    Epoch 79/100
    241/241 - 2s - loss: -7.2032e-01 - custom_acc: 0.7217 - val_loss: -7.0601e-01 - val_custom_acc: 0.7070 - 2s/epoch - 6ms/step
    Epoch 80/100
    241/241 - 2s - loss: -7.2031e-01 - custom_acc: 0.7216 - val_loss: -7.0659e-01 - val_custom_acc: 0.7077 - 2s/epoch - 6ms/step
    Epoch 81/100
    241/241 - 2s - loss: -7.2062e-01 - custom_acc: 0.7219 - val_loss: -7.0636e-01 - val_custom_acc: 0.7076 - 2s/epoch - 7ms/step
    Epoch 82/100
    241/241 - 2s - loss: -7.2090e-01 - custom_acc: 0.7226 - val_loss: -7.0643e-01 - val_custom_acc: 0.7075 - 2s/epoch - 6ms/step
    Epoch 83/100
    241/241 - 2s - loss: -7.2119e-01 - custom_acc: 0.7224 - val_loss: -7.0624e-01 - val_custom_acc: 0.7070 - 2s/epoch - 6ms/step
    Epoch 84/100
    241/241 - 2s - loss: -7.2040e-01 - custom_acc: 0.7219 - val_loss: -7.0591e-01 - val_custom_acc: 0.7070 - 2s/epoch - 6ms/step
    Epoch 85/100
    241/241 - 2s - loss: -7.2066e-01 - custom_acc: 0.7220 - val_loss: -7.0648e-01 - val_custom_acc: 0.7073 - 2s/epoch - 6ms/step
    Epoch 86/100
    241/241 - 2s - loss: -7.2137e-01 - custom_acc: 0.7228 - val_loss: -7.0682e-01 - val_custom_acc: 0.7081 - 2s/epoch - 6ms/step
    Epoch 87/100
    241/241 - 2s - loss: -7.2132e-01 - custom_acc: 0.7227 - val_loss: -7.0636e-01 - val_custom_acc: 0.7077 - 2s/epoch - 6ms/step
    Epoch 88/100
    241/241 - 2s - loss: -7.2089e-01 - custom_acc: 0.7221 - val_loss: -7.0651e-01 - val_custom_acc: 0.7078 - 2s/epoch - 7ms/step
    Epoch 89/100
    241/241 - 2s - loss: -7.2140e-01 - custom_acc: 0.7228 - val_loss: -7.0677e-01 - val_custom_acc: 0.7084 - 2s/epoch - 6ms/step
    Epoch 90/100
    241/241 - 2s - loss: -7.2119e-01 - custom_acc: 0.7227 - val_loss: -7.0694e-01 - val_custom_acc: 0.7088 - 2s/epoch - 6ms/step
    Epoch 91/100
    241/241 - 1s - loss: -7.2145e-01 - custom_acc: 0.7230 - val_loss: -7.0685e-01 - val_custom_acc: 0.7080 - 1s/epoch - 6ms/step
    Epoch 92/100
    241/241 - 2s - loss: -7.2160e-01 - custom_acc: 0.7230 - val_loss: -7.0708e-01 - val_custom_acc: 0.7079 - 2s/epoch - 6ms/step
    Epoch 93/100
    241/241 - 2s - loss: -7.2172e-01 - custom_acc: 0.7231 - val_loss: -7.0689e-01 - val_custom_acc: 0.7077 - 2s/epoch - 6ms/step
    Epoch 94/100
    241/241 - 2s - loss: -7.2193e-01 - custom_acc: 0.7235 - val_loss: -7.0691e-01 - val_custom_acc: 0.7083 - 2s/epoch - 8ms/step
    Epoch 95/100
    241/241 - 2s - loss: -7.2179e-01 - custom_acc: 0.7232 - val_loss: -7.0703e-01 - val_custom_acc: 0.7090 - 2s/epoch - 8ms/step
    Epoch 96/100
    241/241 - 2s - loss: -7.2233e-01 - custom_acc: 0.7239 - val_loss: -7.0695e-01 - val_custom_acc: 0.7089 - 2s/epoch - 8ms/step
    Epoch 97/100
    241/241 - 2s - loss: -7.2169e-01 - custom_acc: 0.7232 - val_loss: -7.0691e-01 - val_custom_acc: 0.7084 - 2s/epoch - 7ms/step
    Epoch 98/100
    241/241 - 2s - loss: -7.2186e-01 - custom_acc: 0.7234 - val_loss: -7.0691e-01 - val_custom_acc: 0.7085 - 2s/epoch - 8ms/step
    Epoch 99/100
    241/241 - 2s - loss: -7.2206e-01 - custom_acc: 0.7236 - val_loss: -7.0661e-01 - val_custom_acc: 0.7080 - 2s/epoch - 7ms/step
    Epoch 100/100
    241/241 - 2s - loss: -7.2217e-01 - custom_acc: 0.7234 - val_loss: -7.0726e-01 - val_custom_acc: 0.7081 - 2s/epoch - 10ms/step



```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(history.history["custom_acc"],label="train",ls="dashed",c="blue")
ax1.plot(history.history['val_custom_acc'],label="test",c="blue")
ax1.set_ylabel('Metric acc',c="blue")

ax2.plot(history.history["val_loss"], label="test",c="green")
ax2.plot(history.history["loss"], label="train",ls="dashed",c="green")
ax2.set_ylabel('Loss = - Metric acc',c="green")

ax1.set_title("L_rate = "+str(L_rate)+" no PCA. Max test acc = "+str(max(history.history['val_custom_acc'])))

ax1.axhline(0.74,c="black",ls="dashdot",alpha=0.5)
ax1.axhline(0.745,c="black",ls="dashdot",alpha=0.5)
ax1.axhline(0.75,c="black",ls="dashdot",alpha=0.5)
ax1.axvline(history.history["val_custom_acc"].index(max(history.history["val_custom_acc"])),c="black",ls="dashdot",alpha=0.5)

# ax1.set_yscale("log")
# axs[i].set_yscale("log")

ax1.set_xlabel('epoch')
plt.legend( loc='center right')
plt.show()
```


    
![png](acroscarrillo_work_files/acroscarrillo_work_140_0.png)
    



```python
def predict_data_to_test(x_test_df,supp_df,model):
    df_x_test = pd.DataFrame()
    for target_ID in x_test_df["ID_TARGET"].unique():
        df_temp_x =  weight_returns(x_test_df,supp_df,target_ID)
        df_x_test = pd.concat([df_x_test,df_temp_x])

    df_x_test =  df_x_test.sort_values(by=['ID'])
    IDs = df_x_test["ID"]
    df_x_test = df_x_test.drop(columns=["ID"]) # unecessary now

    prediction_df = pd.DataFrame()
    prediction_df["ID"] = IDs
    prediction_df["RET_TARGET"] = np.sign(model.predict(df_x_test))

    return prediction_df
    
model.load_weights(checkpoint_filepath)
prediction_df = predict_data_to_test(X_test, X_supp, model)
print("mean ret = "+str(prediction_df["RET_TARGET"].mean()))
prediction_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>RET_TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>267100</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>267101</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>267102</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>267103</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>267104</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>114463</th>
      <td>381563</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114464</th>
      <td>381564</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114465</th>
      <td>381565</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114466</th>
      <td>381566</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>114467</th>
      <td>381567</td>
      <td>-1.0</td>
    </tr>
  </tbody>
</table>
<p>114468 rows × 2 columns</p>
</div>




```python
import string
import random

random_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
print(random_name)
prediction_df.to_csv("./submission_" + str(random_name) + ".csv",index=False)
```

    YX9S


Let's see if we can get any improvement if we apply PCA on the data first: 


```python
model = keras.Sequential([
    tf.keras.layers.InputLayer(150),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dropout(.2),
    # layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

L_rate = 1e-5
model.compile(loss=custom_loss,
            optimizer=tf.keras.optimizers.legacy.Adam(L_rate), metrics=[custom_acc])

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=custom_acc,
    mode='max',
    save_best_only=True
    )


pca = PCA(n_components=150) # explain % of variability
pca.fit( df_x_train )
df_x_train_pca = pca.transform(df_x_train)


## if comented it's because I wanted to delete the verbose output of the fitter.
history_2 = model.fit(df_x_train_pca, df_y_train, validation_split = 0.1, epochs=500,verbose=2,batch_size=32, callbacks=[model_checkpoint_callback])

fig, ax1 = plt.subplots()
axs[i] = ax1.twinx()

ax1.plot(history_2.history["custom_acc"],label="train",ls="dashed",c="blue")
ax1.plot(history_2.history['val_custom_acc'],label="test",c="blue")
ax1.set_ylabel('Metric acc',c="blue")

axs[i].plot(history_2.history["val_loss"], label="test",c="green")
axs[i].plot(history_2.history["loss"], label="train",ls="dashed",c="green")
axs[i].set_ylabel('Loss = - Metric acc',c="green")

ax1.set_title("L_rate = "+str(L_rate)+" with PCA")

# ax1.set_yscale("log")
# axs[i].set_yscale("log")

ax1.set_xlabel('epoch')
plt.legend( loc='center right')
plt.show()
```


```python

```


```python
def gather_data(X_train,supp_df,liquid_ID):
    # filter for liquid_ID
    train_df = X_train.loc[X_train['ID_TARGET'] == liquid_ID] 
    # handle NaNs
    train_df = train_df.fillna(train_df.mean())
    
    cols_names = train_df.columns
    ret_names = []
    for name in cols_names:
        if "RET_" in name:
            ret_names.append(name)

    liquid_classes = supp_df.loc[supp_df['ID_asset'] == liquid_ID].drop(columns=["ID_asset"]).to_numpy()[0] # 4D numpy vector of classes

    temp_df = pd.DataFrame({"ID":train_df["ID"]})
    for ret_name in ret_names: # plus all the other data
        for j in range(4):
            col_name =  ret_name + "_" + str(j)
            iliquid_classes = supp_df.loc[supp_df['ID_asset'] == int(ret_name[4:])].drop(columns=["ID_asset"]).to_numpy()[0]
            
            temp_df[col_name] =  train_df[ret_name].to_numpy() / (1 + np.abs(iliquid_classes[j] - liquid_classes[j]))**2
    return temp_df

```




```python

```


```python

```

# Conclusions
In this notebook, we have developed a few models to predict the returns of liquid assets from available information on related iliquid assets. We first saw how a linear model, helped by a resonable previous feature engenieering, led to a predictive power of up to $73\%$, which is around $3\%$ above the proposed benchmark solution (which is also based on a linear model). Inspired by the success of our feature engenieering and noting that it effectively enlarges the amount of observations from around $3000$ (too little for deep learnign) to around $30.000$ (sufficient for deep learning), we tried out a couple of deep learning models which were able to squeeze an extra $1\%$-$2\%$. This left our top models less than $0.5\%$ away from the best submissions in the leaderboard. This leads me to believe a score of $75\%$ is within reach of this line of progress but unfortunately I dont have enough time to better tune the models or try new ones. 

What a fun challenge!
