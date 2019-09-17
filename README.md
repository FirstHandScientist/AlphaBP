# Alpha Belief Propagation

## Theory Part
For development, please read [analysis for alpha BP](./manuscript/README.md).
For theoretical analysis, please read [manuscript](./manuscript/main.pdf).

## Environment
Create a virtual environment with a python2 interpreter at 'path/to/your/evn/'
```bash
$ virtualenv -p python2.7 pyenv27
```
(Here python2.7 is required due to src/factorgraph.py was developed in python2.7)
Then activate your environment:

``` bash
$ source path/to/your/evn/pyenv27/bin/activate
```
and install the requirement file:

``` bash
$ pip install -r requirements.txt

```

## Experiments

For the sufficient condition check in ER random graph, run

``` bash
$ python bin/contract_condition.py
```
For message change along with iteration numbers, to see case of no convergence

``` bash
$ python bin/converge_rate 0.5 false
```
For message change along with iteration numbers, to see case with convergence

``` bash
$ python bin/converge_rate 0.1 true
```


