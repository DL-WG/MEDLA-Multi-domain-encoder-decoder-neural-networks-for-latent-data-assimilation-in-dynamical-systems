# MEDLA-Multi-domain-encoder-decoder-neural-networks-for-latent-data-assimilation-in-dynamical-systems

## About the project

We introduce a novel deep-learning-based data assimilation scheme, named Multi-domain Encoder-Decoder Latent data Assimilation (MEDLA), capable of handling diverse data sources by sharing a common latent space.
The proposed approach significantly reduces the computational burden since the complex mapping functions are mimicked by the multi-domain encoder-decoder neural network. 
It also enhances assimilation accuracy by minimizing interpolation and approximation errors. Extensive numerical experiments from three different test cases assess MEDLA's performance in high dimensional dynamical 
systems, benchmarking it against state-of-the-art latent data assimilation methods. The numerical results consistently underscore MEDLA's superiority in managing multi-scale observational data and tackling intricate, non-explicit mapping functions.


To test the performance of MEDLA in comparison with the state-of-the-art latent data assimilation approaches, three numerical experiments are designed in this work. The first one involves solving the two-dimensional Burgers' equation
on squared meshes, where both states and observations are the velocity field on a different scale. MEDLA is compared against LA in terms of assimilation accuracy with different levels of observation errors. The second test involves CFD simulations of a multiphase flow 
systems in a pipe with two non-linear state-observation transformation functions. The last test case involves  drop interactions in a microfluidics device with multi-modal data comprising CFD and camera observations, for which no explicit transformation function could be identified.


We provide two implementations of MEDLA in Python using the two most widely adopted deep learning libraries, namely TensorFlow and PyTorch. The numerical experiments of the 2D Burgers' equation and the shallow water models are performed using Tensorflow while the experiments of microfluidic drop interactions are done with Pytorch. 

## Getting Started

Programming language: Python (3.5 or higher)


*   Tensorflow version

| Package Requirement                        |
|--------------------------------------------|
| os                                         |
| sympy                                      |
| numpy                                      |
| pandas                                     |
| math                                       |
| matplotlib                                 |
| Tensorflow (2.3.0 or higher)               |
| Keras (2.4.0 or higher)                    |

*   Pytorch version

| Package Requirement                        |
|--------------------------------------------|
| os                                         |
| numpy                                      |
| pandas                                     |
| math                                       |
| matplotlib                                 |
| Pytorch (2.0.0 or higher)                  |

The main difference between MEDLA and current latent data assimilation methods consists of the multi-domain encoder-decoder as shown in the figure below

![all_LA](https://github.com/DL-WG/MEDLA-Multi-domain-encoder-decoder-neural-networks-for-latent-data-assimilation-in-dynamical-systems/assets/28357071/5b82813e-061f-4315-be48-79cdb302086d)

The entire pipeline of MEDLA with reduced order predictive model is depicted in the following flowchart

![flowchart_MEDLA](https://github.com/DL-WG/MEDLA-Multi-domain-encoder-decoder-neural-networks-for-latent-data-assimilation-in-dynamical-systems/assets/28357071/d7bd516a-c5d2-4d12-a746-d1c66bffa3eb)

