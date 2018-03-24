# Distributed GPU Computing Team
[![Cornell Data Science Logo](images/CDS-banner.png)](cornelldata.science)

Members: Dae Won Kim, Eashaan Kumar, Katarina Jankov

## Project introduction

Deep learning is an exciting field that is applicable to a wide variety of machine learning problems, most notably in areas of computer vision and natural language processing. However, the complexity of neural networks pose a significant challenge in terms of implementation because of the large number of parameters and matrix/vector computations. GPU computing has become the new norm in these computations, but come with difficulties in parallelization, particularly in the multi-node setting. Typically in such a setting, in order to reduce network load and avoid consequent bottlenecks, one must tend toward large batches - as is the approach typically employed by popular batch processing frameworks like hadoop and spark. This, however, comes with a cost to training accuracy. Particularly, the non-convex loss surfaces of neural networks make batch size a very important hyperparameter, and large batches typically lead to convergence to poor local minima. Our objective in this project is to experiment with training relatively large neural network architectures over considerably large clusters using cloud instances. 

## Project Scope

We intend to look into the following: 
- __Tensorflow__

Tensorflow is the most popular execution engine for neural networks. While PyTorch is also an option (as is the whole host of platforms like Caffe2, Keras, Theano and Lasagne), tensorflow is typically easier to expand to larger scales and provides a good starting point for learning and baseline formulations. There also seems to be more widespread efforts to distribute tensorflow over multiple machines - partly due to its more static graph construction. Our aim is to gain relatively wide and thorough understanding of these efforts and their implications, while potentially expanding these efforts to other platforms. 

- [__TensorflowOnSpark__](https://docs.databricks.com/applications/deep-learning/tensorflow.html)

Databrick's convenient documentation: [link](https://docs.databricks.com/applications/deep-learning/tensorflow.html)

This is a library that launches multiple instances of tensorflow using Apache Spark. This is a naive approach to parallelization, and is most useful for hyperparameter optimization. A potential use case may be to join these efforts with SigOpt, which provides hyperparameter optimization on Spark via Bayesian optimization. However, this will not be the main area of interest for us, because while it is helpful to gain an understanding on how it works, it is not a topic that requires intensive research nor is it a field we are capable of making significant contributions to. 

- [__Horovod__](https://github.com/uber/horovod)

 This is a library that is executable on both Keras and Tensorflow, and allows parallelization over multiple GPUs. While Tensorflow also has modules for distributions, Norovod claims that its libraries are much easier to use and achieves better results. We intend to thoroughly understand and experiment with their APIs.

- __Facebook’s paper__: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

This paper claims to have trained a ResNet-50 under 1 hour on 256 GPU instances while maintaining comparable results using very large batch sizes. We wish to replicate the results of this paper (at least validate its core assumption that linearly scaling the learning rate and introducing a “warmup” phase with small learning rates can effectively counteract the setbacks in using large batch sizes. 

