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
```

## TODO

- Implement updateGradInput for > 1D tensors
- Implement Equation (14) for better complexity
- Write more tests
