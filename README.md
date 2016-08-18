## SparseMax in Torch

Implementation of "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification" André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068) in Torch

## SparseMax
Applies the `Sparsemax` function to an n-dimensional input tensor, resulting in an n-dimensional output Tensor whose elements lie in the range (0,1) and sum to 1.

Compared to the `Softmax` function, `Sparsemax` is more likely to contain zero elements.

`Sparsemax` is defined as `f(z)` = `argmin_p ||p - z||^2`, where `p` lies on the `n-1` dimensional probability simplex. In other words, `Sparsemax` is the Euclidean projection of a vector onto the probability simplex. If `z` corresponds to unnormalized event scores then it is likely for the projection to be on the boundary of the simplex, in which case `f(z)` is sparse.

## SparseMaxLoss
Applies the `SparseMaxLoss` function to an n-dimensional tensor.

The `SparseMaxLoss` function defines a loss for `Sparsemax` that is similar to the `LogSoftMax` function for `Softmax` in that it defines a loss function whose gradient takes the same form as that of `LogSoftMax`. That is the gradient of `SparseMaxLoss(z,y)` is `y - sparsemax(z)` where `y` is a one-hot encoding of the correct target.

Because of its similar form, the `SparseMaxLoss` function can be combined with the `ClassNLLCriterion` criterion to train an `nn` model. Note however that the `SparseMaxLoss` is not the negative log-likelihood of the sparsemax distribution, which would be undefined on the zero elements, and does not have the same interpretation as `LogSoftMax`.

## SparseMaxCriterion
This criterion combines `SparseMaxLoss` and `ClassNLLCriterion` into one class, similar to `CrossEntropyCriterion`.

The loss can be described as `loss(x, class) = -SparseMaxLoss(x)[class]`.

The losses are average across observations for each minibatch.

## Examples

```lua

th> x = torch.randn(5)
th> y = torch.Tensor({3})

th> x
 0.8053
 0.4594
-0.6136
-0.9460
 1.0722
[torch.DoubleTensor of size 5]

th> nn.SoftMax():forward(x)
 0.2916
 0.2064
 0.0706
 0.0506
 0.3808
[torch.DoubleTensor of size 5]

th> nn.SparseMax():forward(x)
 0.3597
 0.0138
 0.0000
 0.0000
 0.6265
[torch.DoubleTensor of size 5]

...
======= SparseMaxLoss 1D Input Test =======
Input x:
-0.3218
 0.7395
-0.2319
 0.2312
 0.7185
[torch.DoubleTensor of size 5]

Target:
 3
[torch.DoubleTensor of size 1]

LogSoftMax x:
-2.2569
-1.1955
-2.1669
-1.7038
-1.2165
[torch.DoubleTensor of size 5]

SparseMaxLoss x:
-1.2482
-0.1869
-1.1582
-0.6951
-0.2078
[torch.DoubleTensor of size 5]

LogSoftMax+NLL, CrossEntropyCriterion:
2.1669343084448	2.1669343084448
SparseMaxLoss+NLL, SparseMaxCriterion:
1.1582469533367	1.1582469533367
NLL Gradient for Softmax:
 0
 0
-1
 0
 0
[torch.DoubleTensor of size 5]

NLL Gradient for Sparsemax:
 0
 0
-1
 0
 0
[torch.DoubleTensor of size 5]

LogSoftMax Gradient, CrossEntropyCriterion Gradient:
 0.1047
 0.3025
-0.8855
 0.1820
 0.2963
[torch.DoubleTensor of size 5]

 0.1047
 0.3025
-0.8855
 0.1820
 0.2963
[torch.DoubleTensor of size 5]

SparseMaxLoss Gradient, SparseMaxCriterion Gradient:
 0.0000
 0.5097
-1.0000
 0.0015
 0.4888
[torch.DoubleTensor of size 5]

 0.0000
 0.5097
-1.0000
 0.0015
 0.4888
[torch.DoubleTensor of size 5]

SoftMax x:
 0.1047
 0.3025
 0.1145
 0.1820
 0.2963
[torch.DoubleTensor of size 5]

SparseMax x (from loss backward):
 0.0000
 0.5097
 0.0000
 0.0015
 0.4888
[torch.DoubleTensor of size 5]
```
## TODO

- SparseMax forward pass uses sort(input) to find threshold but this can be replaced with an expected linear time selection algorithm.
- Use sparse tensors/data structures to take advantage of sparsity for very high dimensional inputs.
- Optimized THNN implementation.
