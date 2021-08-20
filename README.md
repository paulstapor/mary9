# Mary9
A python toolbox for hybrid global-local optimization

## About
Mary9 is a toolbox for optimization of real valued, potentially 
high-dimensional, non-convex, and multi-modal objective functions.
It aims at combining the strengths of two different optimization concepts:
the exploration capacities of gradient-free, global optimization algorithms and
the (typically superior) exploitation possibilities of gradient-based, 
(multi-start) local optimization approaches. Mary9 is supposed to work similar 
to the MATLAB-based hybrid global-local optimization toolbox 
[MEIGO](http://gingproc.iim.csic.es/meigo.html "MEIGO toolbox"), but 
implements some additional ideas, such as the concept of 
[hyper-heuristics](https://link.springer.com/article/10.1007/s11227-019-02871-0
"Research paper on hyper-heuristics"),
or the more recent local optimization toolbox Fides.

Mary9 is named after the space probe *Mariner 9*, which was sent to Mars in 
1971. It discovered the Mariner Valleys (Valles Marineris), the largest and 
deepest valleys on the Solar System, and hence the currently deepest valleys
known to mankind, i.e., *the* global minimum per se.


## Motivation
This is my personal "parental leave/I got some leisure time/I find that 
useful"-project, no guarantees on whatsoever. However, I think that something
like this as python toolbox is useful, as MEIGO is/was widely use in MATLAB,
but there's no real equivalent in python.
In, e.g., the field of parameter estimation of ODE models, multi-start local
optimization is widely used, as cost functions are often smooth and somewhat
multi-modal, but often not multi-modal enough to do really harm. Hence, 
exploitation is more important than exploration, and using a good local 
optimizer with some random initialization and sufficiently many starts is 
usually good enough. However, there are *some* models (e.g., if the 
underlying ODE shows oscillatory behavior), where multi-start local 
optimization fails.

Generally, Mary9 will be suited for optimizing non-convex, multi-modal 
problems, in which evaluating the objective function is computationally 
rather expensive (at least substantially more expensive than the proper time 
of the optimizer itself).


## Algorithm and Concept
Mary9 works with a set of global optimization algorithms, which are both 
initialized with populations based on creating a large latin hypercube 
sample and keeping a set of samples which are particularly fit and diverse.
Currently only two algorithms are implemented: the CMAES algorithm from the 
cma package, and the Differential Evolution algorithm from scipy.optimize).
However, extensions by a scatter search and a particle swarm algorithm are 
planned.
Some remaining points from the initial proposals (for both optimizers) are 
kept, and short local optimizations are run from them (this is called the 
"initial refinement" in the code). If promising points in parameter space are 
found, the populations of the global optimizers are updated from those.

Then, the global exploration phase starts and the global optimizers are run 
for a couple of iterations. Again, if promising points in parameter space 
are found, the global optimizers will exchange these each time, in order to 
progress faster.

A certain number of times, roughly in the middle of this global exploration 
phase, the populations of the global optimizers are merged and subpopulations
are selected from this larger population. These subpopulations will be used to 
run short local optimizations again (called "runtime refinements" in the code).
The individuals in the populations of the global optimizers are then replaced 
with these "refined" individuals. Currently, there are three such runtime 
refinement phases, but it is planned to implement this in a more customizable 
and flexible way.

At the end of the global exploration phase, the populations of the global 
optimizers are merged again to create on large "final population".
Again, a subsample is selected from this final population and long local 
optimization runs are launched from those in the spirit of a multi-start local 
optimization. The results of this local exploitation phase are then sorted and 
returned to the user, together with the final population. 

## Documentation
There will be some documentation on readthedocs in the near future.
