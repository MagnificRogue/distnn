distnn = require 'distnn'
torch = require 'torch'


--example of creating a new input object
labels = torch.Tensor(100)

input = distnn.Input.new(torch.Tensor(32,32,3,100):zero(),torch.Tensor(100):zero())
print(input)

mpi.finalize()
