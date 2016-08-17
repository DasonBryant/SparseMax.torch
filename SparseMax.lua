require 'nn'

-- Implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)

local SparseMax = torch.class('nn.SparseMax', 'nn.Module')

function SparseMax:updateOutput(input)
  local dim = 1
  local inputDim = input:nDimension()
  if inputDim == 2 or inputDim == 4 then -- match functionality of nn.SoftMax
    dim = 2
  elseif input:nDimension() > 4 then
    error('1D, 2D, 3D or 4D tensor expected')
  end
  local sizeDim = input:size()[dim]

  -- Sort input in descending order.
  -- (NOTE: Can be replaced with linear time selection method described here:
  --  http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
  local zs = torch.sort(input, dim, true)

  local range = torch.range(1, sizeDim):typeAs(input)
  if dim == 2 then
    range = range:view(1, sizeDim)
  end
  range = range:expandAs(zs)

  -- Determine sparsity of projection
  local bound = 1 + torch.cmul(range, zs)
  local cumsum_zs = torch.cumsum(zs, dim)
  local is_gt = torch.gt(bound, cumsum_zs):typeAs(range)
  local k = torch.max(torch.cmul(range, is_gt), dim)

  -- Compute threshold function
  local zs_sparse = torch.cmul(is_gt, zs)
  local taus = torch.cdiv(torch.sum(zs_sparse, dim) - 1, k)

  -- Sparsemax
  self.output = torch.cmax(
    torch.zeros(input:size()):typeAs(input),
    input - taus:expandAs(input)
  )
  return self.output
end

function SparseMax:updateGradInput(input, gradOutput)
  local dim = 1
  local inputDim = input:nDimension()
  if inputDim == 2 or inputDim == 4 then
    dim = 2
  elseif input:nDimension() > 4 then
    error('1D, 2D, 3D or 4D tensor expected')
  end
  local sizeDim = input:size()[dim]

  local nonzeros = torch.ne(self.output, 0):typeAs(self.output)
  self.gradInput = gradOutput - torch.sum(torch.cmul(gradOutput,nonzeros),dim):expandAs(gradOutput)
  self.gradInput = torch.cmul(nonzeros, self.gradInput)
  return self.gradInput
end
