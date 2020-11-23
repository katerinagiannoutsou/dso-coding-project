# DSO-670: Coding Project

Additional numerical experiments on shortest path problem using the methods discussed in the paper: 
"Smart 'Predict, Then Optimize'" by Adam Elmachtoub and Paul Grigas.
https://arxiv.org/abs/1710.08005 

https://github.com/paulgrigas/SmartPredictThenOptimize

# Overview

sanity_check : basic sanity check to familiarize with code from original paper

Following structure of code from the authors' repository:

solver: contains reformulation and sgd method needed to run the SPO+, and other linear objective methods, and random forest training

oracles: contains the shortest path oracle

experiments: running the experiment and populating results' table

plots: plots contained in project
