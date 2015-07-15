local mpi = require("mpi")
local torch = require("torch")

Input  = {}

Input.__index = Input

function Input.new(data, labels, size)
  local self = setmetatable({}, Input)
  self.data = data
  self.labels = labels
  self.size = size
  return self
end


return Input
