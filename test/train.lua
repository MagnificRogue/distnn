require("torch")
require ("nn")
local mpi = require ("mpi")
local ffi = require ("ffi")
local distnn = require("distnn")


comm = mpi.comm_world
rank = comm:rank()
size = comm:size()

--some input parameters you can tweak easily
height = 16
width = 16
channels = 1
nTrain = 10
nTest = 8


local node_trainingData = {}


if rank == 0 then


  dofile './generate_testing_data.lua'
  dofile './generate_training_data.lua'


  --rank 0 chunks train data/labels and places in buffer

  ---Lets send the data to each process!


  for key, subbatch in ipairs(trainData.data:chunk(size,1))  do
    
    local target_node = key-1

    local tensor = torch.Tensor(subbatch:size()):copy(subbatch)


    if target_node ~= 0 then
      print("Rank 0 is sending data chunk to " .. target_node)
      distnn.mpi_extensions.send_tensor(tensor, target_node, 1, mpi.comm_world)
 
    else

     node_trainingData.data = tensor

    end
  end


  for key, subbatch_labels in ipairs(trainData.labels:chunk(size,1)) do

    local target_node = key-1

    local tensor = torch.Tensor(subbatch_labels:size()):copy(subbatch_labels)


    if target_node ~= 0 then
      print("Rank 0 is sending label chunk to " .. target_node)
      distnn.mpi_extensions.send_tensor(tensor, target_node, 2, mpi.comm_world)
  
    else

      node_trainingData.labels = tensor

    end

  end


else

  node_trainingData.data = distnn.mpi_extensions.receive_tensor(4, 0, 1, mpi.comm_world)
  node_trainingData.labels = distnn.mpi_extensions.receive_tensor(1, 0, 2, mpi.comm_world)
  
end

  node_trainingData.size = function() return (#node_trainingData.labels)[1] end
  
  print(node_trainingData)
  print(rank .. " is at the barrier")
  mpi.barrier(mpi.comm_world)
  --at this point, all the data is in node_trainingData



--[[
  --------------------------------------------------
  --DISTRIBUTE THE MODEL----------------------------
  --------------------------------------------------
--]]

--create a model on each node
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 3, 3, 3))
model:add(nn.SpatialConvolutionMM(3, 4, 3, 3))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(4*6*6))
model:add(nn.Linear(4*6*6, 128))
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())
print("Here is the model from rank " .. rank)
print(model)
criterion = nn.ClassNLLCriterion()

--extract parameters (and gradients) from each node's model
--(we'll need to access the gradients during training)
local parameters = model:getParameters()

----rank 0 sends its initial parameters to each process
if rank == 0 then
 
  for dest = 1, size-1 do
     distnn.mpi_extensions.send_tensor(parameters, dest, 11, mpi.comm_world)
  end
  
else
  
  --copy rank 0's parameters into this node's model
  parameters:copy(distnn.mpi_extensions.receive_tensor(1, 0, 11, mpi.comm_world))

end

mpi.barrier(mpi.comm_world)
--at this point, each node has a model with the same weights and biases


--[[
  --------------------------------------------------
  --TRAINING----------------------------------------
  --------------------------------------------------
--]]

local learningRate = 0.01
local max_epochs = 1

for epoch = 1, max_epochs do
 

  print("Training on epoch 1 rank " .. rank)

  --shuffle the indexes of your training samples
  shuffle = torch.randperm(node_trainingData:size())


  for t = 1, node_trainingData:size() do

    --load sample
    local input = node_trainingData.data[shuffle[t]]
    local target = node_trainingData.labels[shuffle[t]]
    --reset gradients
    model:zeroGradParameters()        
    --feed forward through model
    local output = model:forward(input)
    --calculate error
    local err = criterion:forward(output, target)
    --calculate initial gradients at output layer
    local gradInput = criterion:backward(output, target)
    --perform backpropagation 
    model:updateGradInput(input, gradInput)
    model:accUpdateGradParameters(input, gradInput, learningRate)

    ------Average the weights across all models 
    
    --first a sum reduction to add weights across models
    parameters = distnn.mpi_extensions.allreduce_tensor(parameters, mpi.sum, mpi.comm_world)
    --then each model divides its weights by the number of models
    parameters:copy(parameters / size)
    --the network parameters should be consistant across models now
  end

end

---training is complete

mpi.finalize()




