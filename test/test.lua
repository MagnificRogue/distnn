distnn = require 'distnn'
torch = require 'torch'


--example of creating a new input object
input = distnn.Input.new(torch.Tensor(1),torch.Tensor(2),3)
