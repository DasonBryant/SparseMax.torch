require 'nn'
require 'cunn'
require 'SparseMax'
require 'SparseMaxLoss'
require 'SparseMaxCriterion'

function testSparseMax(cuda)
    local cuda = cuda or false
    local dtype = nil

    if cuda then
        dtype = 'torch.CudaTensor'
    else
        dtype = 'torch.DoubleTensor'
    end

    local a1 = torch.rand(10):type(dtype)
    local a2 = torch.rand(2,10):type(dtype)
    local a3 = torch.rand(4,2,2):type(dtype)
    local a4 = torch.rand(2,5,2,2):type(dtype)

    local soft = nn.SoftMax():type(dtype)
    local sm = nn.SparseMax()

    print('======= SparseMax 1D Input Test =======')
    print('-> 1D Input')
    print(a1)
    print('-> SoftMax')
    print(soft:forward(a1))
    print('-> SparseMax')
    print(sm:forward(a1))

    print('\n\n======= SparseMax 2D Input Test =======')
    print('-> 2D Input')
    print(a2)
    print('-> SoftMax')
    print(soft:forward(a2))
    print('-> SparseMax')
    print(sm:forward(a2))
  end

function testSparseMaxLoss(x,y)
  local lsm = nn.LogSoftMax()
  local sml = nn.SparseMaxLoss()
  local nll = nn.ClassNLLCriterion()
  local cec = nn.CrossEntropyCriterion()
  local smc = nn.SparseMaxCriterion()
  print('Input x:')
  print(x)
  print('Target:')
  print(y)

  local x_lsm = lsm:forward(x)
  local x_sml = sml:forward(x)
  print('LogSoftMax x:')
  print(x_lsm)
  print('SparseMaxLoss x:')
  print(x_sml)

  local lsm_xent = nll:forward(x_lsm, y)
  local sml_crit = nll:forward(x_sml, y)
  cec:forward(x, y)
  smc:forward(x, y)
  print('LogSoftMax+NLL, CrossEntropyCriterion:')
  print(lsm_xent, cec.output)
  print('SparseMaxLoss+NLL, SparseMaxCriterion:')
  print(sml_crit, smc.output)

  local lsm_nll_grad = nll:backward(x_lsm, y)
  local sml_nll_grad = nll:backward(x_sml, y)
  print('NLL Gradient for Softmax:')
  print(lsm_nll_grad)
  print('NLL Gradient for Sparsemax:')
  print(sml_nll_grad)

  local lsm_grad = lsm:backward(x, lsm_nll_grad)
  local sml_grad = sml:backward(x, sml_nll_grad)
  print('LogSoftMax Gradient, CrossEntropyCriterion Gradient:')
  print(lsm_grad, cec:backward(x,y))
  print('SparseMaxLoss Gradient, SparseMaxCriterion Gradient:')
  print(sml_grad, smc:backward(x,y))

  print('SoftMax x:')
  print(nn.SoftMax()(x))
  print('SparseMax x (from loss backward):')
  print(sml.sparsemax)
end

testSparseMax()

local x = torch.randn(5)
local y = torch.Tensor({3})
print('======= SparseMaxLoss 1D Input Test =======')
testSparseMaxLoss(x,y)

local x = torch.randn(2,4)
local y = torch.Tensor({1,4})
print('======= SparseMaxLoss 2D Input Test =======')
testSparseMaxLoss(x,y)
