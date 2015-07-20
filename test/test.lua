distnn = require 'distnn'
torch = require 'torch'


--example of creating a new input object
labels = torch.Tensor(100)

t = torch.Tensor(32,32,3,100):zero()

t[1][1][1][1] = 5

input = distnn.Input.new(t,torch.Tensor(100):zero())


for key,value in pairs(input.distributed_data) do 
  if(key == 0) then
    print(value:select(1,1))
  end
end


mpi.finalize()
