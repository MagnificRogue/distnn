local mpi = require("mpi")
local torch = require("torch")

Input  = {}

Input.__index = Input

function Input.new(data, labels, input_count)
  local self = setmetatable({}, Input)

  assert(getmetatable(data) == getmetatable(torch.Tensor))
  assert(getmetatable(labels) == getmetatable(torch.Tensor))
  assert(getmetatable(input_count) == nil)

  self.data = data
  self.labels = labels
  self.size = input_count

  self:distribute()

  return self
end

function Input.distribute()




end

return Input
