local torch = require 'torch'
local mpi = require 'mpi'

local distnn = {}

distnn.hellomodule = require 'distnn.hellomodule'
distnn.Input = require 'distnn.Input'






torch.Tensor.mpi = function(self)
print("IT WORKED")
print(self)
print("IT WORKED")
end

return distnn
