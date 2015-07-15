local mpi = require("mpi")
local torch = require("torch")

Input  = {}

Input.__index = Input

function Input.new(data, labels, size)
  local self = setmetatable({}, Input)

  assert(getmetatable(data) == getmetatable(torch.Tensor))
  assert(getmetatable(labels) == getmetatable(torch.Tensor))
  assert(getmetatable(size) == nil)

  self.data = data
  self.labels = labels
  self.size = size
  return self
end


return Input
