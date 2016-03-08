require 'nn'

-- Implementation attempt of "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- paper (André F. T. Martins, Ramón Fernandez Astudillo: http://arxiv.org/abs/1602.02068) in Torch

local SparseMax, _ = torch.class('nn.SparseMax', 'nn.Module')

function SparseMax:updateOutput(input)

    local dim = nil
    local inputDim = input:nDimension()

    --for compatibility with SoftMax
    if inputDim == 1 then
        dim = 1
    elseif inputDim == 2 then
        dim = 2
    elseif inputDim == 3 then
        dim = 1
    elseif inputDim == 4 then
        dim = 2
    end

    local bigk = input:size(dim)

    -- Sort the input in descending order
    -- First line of Algorithm 1
    local zs = torch.sort(input, dim, true)

    -- Prepare a repeated range tensor
    local range = torch.range(1, bigk):typeAs(input)
    local rangeViewMask = zs:size():fill(1)
    rangeViewMask[dim] = -1
    range = range:view(rangeViewMask):expandAs(zs)

    -- Element-wise multiply repeated range tensor with sorted input
    local left = torch.cmul(range, zs) + 1
    local right = torch.cumsum(zs, dim)

    local lt = torch.gt(left, right)

    -- Second line of Algorithm 1
    local k = torch.max(torch.cmul(range, lt:typeAs(range)), dim)

    -- Third line of Algorithm 1, also given in Equation (4)
    local threshold = torch.cdiv(torch.sum(torch.cmul(lt:typeAs(zs), zs), dim)-1, k)
    self.output = torch.cmax(torch.zeros(input:size()):typeAs(input), input-threshold:expandAs(input))

    return self.output
end

function SparseMax:updateGradInput(input, gradOutput)

    local dim = nil
    local inputDim = input:nDimension()

    --for compatibility with SoftMax
    if inputDim == 1 then
        dim = 1
    elseif inputDim == 2 then
        dim = 2
    elseif inputDim == 3 then
        dim = 1
    elseif inputDim == 4 then
        dim = 2
    end

    local S = torch.ne(self.output, 0):typeAs(self.output)

    -- Equation (12)
    local J = torch.diag(S) - torch.ger(S,S) / torch.sum(S)

    -- TODO: implement gradient for > 1D tensors
    -- TODO: Implement Equation (14) for nicer complexity

    self.gradInput = J * gradOutput
    return self.gradInput

end

function SparseMax:test(cuda)
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

    print('======= 1D Test =======')
    print('-> 1D Input')
    print(a1)
    print('-> SoftMax')
    print(soft:forward(a1))
    print('-> SparseMax')
    print(sm:forward(a1))

    print('\n\n======= 2D Test =======')
    print('-> 2D Input')
    print(a2)
    print('-> SoftMax')
    print(soft:forward(a2))
    print('-> SparseMax')
    print(sm:forward(a2))

    print('\n\n======= 3D Test =======')
    print('-> 3D Input')
    print(a3)
    print('-> SoftMax')
    print(soft:forward(a3))
    print('-> SparseMax')
    print(sm:forward(a3))

    print('\n\n======= 4D Test =======')
    print('-> 4D Input')
    print(a4)
    print('-> SoftMax')
    print(soft:forward(a4))
    print('-> SparseMax')
    print(sm:forward(a4))
end
