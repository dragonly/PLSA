This is an reimplementation of PLSA forked from [hitalex](https://github.com/hitalex/PLSA).

On finding it too slow for even a small corpus like [BBC Corpus 2006](http://mlg.ucd.ie/datasets/bbc.html), I reimplemented the EM algorithm part of the PLSA model.
Instead of using nested loops in python code, which is super slow, I turn the computation into matrix operations utilizing numpy library functions, which are implemented in C code.

Experiment shows that it is roughly **100** times FASTER!

**WARNNING**: the 5.1M bbc corpus dataset will swallow up about 6 to 8 Gigabytes of RAM, so be careful if you wanna run it directly. I suggest running it on a dedicated server, or cut out some of the corpus documents before running the main.py file.

**Maximum A Posterior (MAP) Estimation**
I added a `priori.conf` file, which contains the priori distribution for a MAP estimation of the PLSA model

the first line contains number of words in the pseudo documents you give as a priori, each line following contains priori distribution for each topic. the line of priori distribution should be agree with `<number_of_topics> you specify in command line.

if the pseudo count (first line of the `priori.conf` is 0, then you're running a normal **Maximum Likelihood Estimation** (MLE) of the PLSA model.

## Instructions
```python
python main.py <number_of_topics> <number_of_iterations>
```
