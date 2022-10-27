# miss_hmm

A package that performs inference and prediction on hidden Markov model (HMM) with missing observations.

Currently, the package **miss_hmm** supports parameter learning and hidden state prediction with incomplete observation sequences.

The inference (learning) procedure is done by EM algorithm. As for hidden state prediction, miss_hmm supports Viterbi decoding, marginalized MAP decoding and random guessing (for comparison).

## Installation

The **miss_hmm** package can be installed via pip command:
```python
!pip install miss_hmm
```

# Inference

The **miss_hmm** supports a model class *hmm_model* for inference. To create an *hmm_model* object, use:

```python
# where hmm_model locates
from miss_hmm.model import hmm_model
model=hmm_model(hidden_state,obs_state,indicator=None)
```

The first parameter *hidden_state* is a **numpy** array that includes all hidden states in the model. The parameter *obs_state* is also a **numpy** array that includes all observable states. Both parameters can be arrays of numbers (int, float, etc) or strings or data structures that have string representations.

The third parameter *indicator* is unique in hidden Markov model with missing observations. The indicator variable specifies which tag you use to identify those missing observations. A *None* tag will be used by default. However, you can specify the missing tag in your database by specifying the *indicator* parameter.

To perform inference, use:
```python
model.fit(data,step=5,e=0.001,core=None)
```
After calling the function *model.fit*, parameters will be learnt automatically. 

The parameter *data* is the dataset for learning. Parameter *step* specifies how often you would like the parameter to print the training log. If *step=5*, then the training log will be printed every 5 iterations. Current estimate, number per iteration and target function will be printed. If you set *step=0*, then no log will be printed.

The paramter *e* specifies the stopping criteria.

The parameter *core* specify the number of CPU cores you want to use (for parallel acceleration). If *core=None*, then *fit()* will automatically make use of **ALL** CPU cores available. The multiprocessing unit is implemented via the Python module *multiprocessing*. Therefore, if your program got stuck while training (especially on Windows platform), putting the *fit()* function into main function will solve the problem.

After training, you can access the estimate by calling:
```python
# view transition matrix
model.transition
# view emission matrix
model.emission
# view initial distribution
model.initial
```
If you further specify the parameter *step* to be a positive number in *fit()*, you can print a training log via:
```python
model.train_log
```
which is a list of string.

## Classification

After learning parameter, one can classify hidden states using available observations. Simply call:
```python
z=model.predict(data,method='Viterbi')
```
Here *data* is the dataset that you want to classify. For *method* parameter, *method='Viterbi'* will provide an MAP estimate according to the observations. You can also use *method='marginalized'* to obtain a marginalized MAP, which will have higher accuracy (but the output hidden path may not always be a legal path) or use *method='random'* to simply produce a random guess.

By default, the *predict()* function produces an MAP (that is, 'Viterbi').

## Simulation

The **miss_hmm** package also supports to simulate a hidden Markov model with missing observations. It is implemented by the *HMM* class. Use:
```python
# where HMM class locates
from miss_hmm.HMM import HMM
chain=HMM(hidden_state,obs_state,transition,obs_prob,pi)
path,data=chain.generate_partial_seq(size=100,length=10,p=0.3)
```
Here *transition, obs_prob, pi* are transition matrix, emission matrix and initial distributions respectively. *HMM()* produces an *HMM* object. The *generate_partial_seq()* function will produce two datasets, the *path* dataset consists of all hidden sequences and the *data* dataset produces observation with missings (missing tag is None). Among the parameters, *size* is the total number of sequences and *length* is the number of each sequence. *p* is the missing rate (entries are set to be missing randomly according to the missing rate).

## A Simulated Example

In this section, we demonstrate how to use the package *miss_hmm* by running the model on a simulated dataset：

```python
import numpy as np
# hmm_model is inside miss_hmm.model
from miss_hmm.model import hmm_model
# HMM class in miss_hmm.HMM
from miss_hmm.HMM import HMM

# specify states and true parameters
transition=np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
state=np.array(['0','1','2'])
hidden_state=state
obs_state=np.array(['Blue','Green','Yellow'])
obs_prob=np.array([[0.9,0.05,0.05],
                   [0.1,0.7,0.2],
                   [0.15,0.05,0.8]
    ])
pi=np.array([0.6,0.3,0.1])

# simulate a dataset
chain=HMM(hidden_state,obs_state,transition,obs_prob,pi)
# path are all true hidden variables, data is the incomplete observation, consisting of 100 sequences, each of length 10, missing rate 0.3
path,data=chain.generate_partial_seq(100,10,0.3)

if __name__=='__main__':
    # initialize the model
    model=hmm_model(hidden_state,obs_state)
    # fit the model
    a,b,c,f,l=model.fit(data,5,0.01)
    # view learnt parameters
    print(model.transision)
    print(model.emission)
    print(model.initial)
    # hidden state classification
    z=model.predict(data)
```

## To Coming...

1. Improve the numerical stalability of training
2. Support user-defined initial value (in learning)
