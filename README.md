## SparseMax in Torch

Implementation attempt of "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification" paper (André F. T. Martins, Ramón Fernandez Astudillo: http://arxiv.org/abs/1602.02068) in Torch

## Example

```lua

  ______             __   |  Torch7
 /_  __/__  ________/ /   |  Scientific computing for Lua.
  / / / _ \/ __/ __/ _ \  |  Type ? for help
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch
                          |  http://torch.ch

th> nn = require 'nn'
                                                                      [0.0000s]
th> a1 = torch.rand(10)
                                                                      [0.0001s]
th> a1
 0.1851
 0.5324
 0.2838
 0.0006
 0.3066
 0.0450
 0.4762
 0.2809
 0.0798
 0.9339
[torch.DoubleTensor of size 10]

                                                                      [0.0001s]
th> nn.SoftMax():forward(a1)
 0.0847
 0.1199
 0.0935
 0.0705
 0.0957
 0.0737
 0.1134
 0.0932
 0.0763
 0.1792
[torch.DoubleTensor of size 10]

                                                                      [0.0058s]
th> nn.SparseMax():forward(a1)
 0.0000
 0.2182
 0.0000
 0.0000
 0.0000
 0.0000
 0.1620
 0.0000
 0.0000
 0.6197
[torch.DoubleTensor of size 10]

th> nn.SparseMax():test()

...

======= 2D Test =======
-> 2D Input
 0.9696  0.2880  0.2762  0.3397  0.2526  0.0734  0.9772  0.7903  0.0893  0.0026
 0.9996  0.5334  0.5426  0.8036  0.2914  0.9608  0.6773  0.5930  0.5310  0.3592
[torch.DoubleTensor of size 2x10]

-> SoftMax
 0.1647  0.0833  0.0823  0.0877  0.0804  0.0672  0.1659  0.1376  0.0683  0.0626
 0.1412  0.0886  0.0894  0.1161  0.0696  0.1359  0.1023  0.0940  0.0884  0.0744
[torch.DoubleTensor of size 2x10]

-> SparseMax
 0.3906  0.0000  0.0000  0.0000  0.0000  0.0000  0.3982  0.2113  0.0000  0.0000
 0.3893  0.0000  0.0000  0.1933  0.0000  0.3505  0.0669  0.0000  0.0000  0.0000
[torch.DoubleTensor of size 2x10]


======= 3D Test =======
-> 3D Input
(1,.,.) =
  0.9208  0.7367
  0.3335  0.3119

(2,.,.) =
  0.8805  0.7559
  0.1164  0.9982

(3,.,.) =
  0.4688  0.6817
  0.7545  0.5034

(4,.,.) =
  0.7565  0.2452
  0.8948  0.9769
[torch.DoubleTensor of size 4x2x2]

-> SoftMax
(1,.,.) =
  0.2902  0.2795
  0.1968  0.1628

(2,.,.) =
  0.2788  0.2849
  0.1584  0.3234

(3,.,.) =
  0.1847  0.2646
  0.2998  0.1972

(4,.,.) =
  0.2463  0.1710
  0.3450  0.3166
[torch.DoubleTensor of size 4x2x2]

-> SparseMax
(1,.,.) =
  0.4015  0.3452
  0.0059  0.0000

(2,.,.) =
  0.3612  0.3645
  0.0000  0.5053

(3,.,.) =
  0.0000  0.2903
  0.4269  0.0106

(4,.,.) =
  0.2372  0.0000
  0.5672  0.4841
[torch.DoubleTensor of size 4x2x2]

...
```

## TODO

- Implement updateGradInput for > 1D tensors
- Implement Equation (14) for better complexity
- Write more tests
