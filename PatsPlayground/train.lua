require("torch")
require ("nn")

mpi = require ("mpi")
ffi = require ("ffi")
dofile './distnn_libb.lua'




comm = mpi.comm_world
rank = comm:rank()
size = comm:size()

--some input parameters you can tweak easily
height = 16
width = 16
channels = 1
nTrain = 100
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

      mpi.send_tensor(tensor, target_node, 1, mpi.comm_world)
 
    else

     node_trainingData.data = tensor

    end
  end


  for key, subbatch_labels in ipairs(trainData.labels:chunk(size,1)) do

    local target_node = key-1

    local tensor = torch.Tensor(subbatch_labels:size()):copy(subbatch_labels)


    if target_node ~= 0 then
      
      mpi.send_tensor(tensor, target_node, 2, mpi.comm_world)
  
    else

      node_trainingData.labels = tensor

    end

  end


else

  node_trainingData.data = mpi.receive_tensor(4, 0, 1, mpi.comm_world)
  node_trainingData.labels = mpi.receive_tensor(1, 0, 2, mpi.comm_world)
  
end

  node_trainingData.size = function() return #labels end

  print(node_trainingData)

  mpi.barrier(mpi.comm_world)
  --at this point, all the data is in node_trainingData


--[[


print("SAFE")

print('barrier '.. rank)
mpi.barrier(comm)

print('hello '.. rank)
--rank 0 broadcasts chunk size info to each node
mpi.bcast(trnSizeinfo, 1, mpi.int, 0, comm)
print("SAFE")
trnSizes = torch.LongStorage(4)
for i=1, 4 do
  trnSizes[i] = trnSizeinfo[i-1]
end

mpi.bcast(tstSizeinfo, 4, mpi.int, 0, comm)
tstSizes = torch.LongStorage(4)
for i=1, 4 do
  tstSizes[i] = tstSizeinfo[i-1]
end

print('rank ' .. rank .. ' size of my chunk: ')
print(trnSizes)
print(tstSizes)

--now that each node has chunk size info, we can make the receiving buffers
local nTrnElem = trnSizes[1] * trnSizes[2] * trnSizes[3] * trnSizes[4]
print('nTrnElem ' .. nTrnElem)
local mytrnDatabuf = ffi.new("double[?]", nTrnElem)

local nTstElem = tstSizes[1] * tstSizes[2] * tstSizes[3] * tstSizes[4]
print('nTrnElem ' .. nTrnElem)
local mytstDatabuf = ffi.new("double[?]", nTstElem)

local mytrnLabelbuf = ffi.new("double[?]", trnSizes[1])
local mytstLabelbuf = ffi.new("double[?]", tstSizes[1])

mpi.barrier(comm)

--scatter operations for train and test data and labels
--rank 0 scatters blocks of fixed size to other nodes
--this assumes the number of nodes divides the training data evenly (we may wanna fix this later to allow variable chunk sizes (it can be done with mpi.scatterv)) 
mpi.scatter(trnDatabuf, nTrnElem, mpi.double, mytrnDatabuf, nTrnElem, mpi.double, 0, comm)
mpi.scatter(trnLabelbuf, trnSizes[1], mpi.double, mytrnLabelbuf, trnSizes[1], mpi.double, 0, comm)
mpi.scatter(tstDatabuf, nTstElem, mpi.double, mytstDatabuf, nTstElem, mpi.double, 0, comm)
mpi.scatter(tstLabelbuf, tstSizes[1], mpi.double, mytstLabelbuf, tstSizes[1], mpi.double, 0, comm)

mpi.barrier(comm)

print('heyy ' .. rank)

--copy all the buffers into the appropriate tensors

--train data
local recov_trnData = torch.DoubleStorage(nTrnElem)
for i=1, nTrnElem do
  recov_trnData[i] = mytrnDatabuf[i-1]
end
t_trnData = torch.Tensor(recov_trnData, 1, #recov_trnData, 1)
t_trnData:view(trnSizes)
--print(#t_trnData)

--test data
local recov_tstData = torch.DoubleStorage(nTstElem)
for i=1, nTstElem do
  recov_tstData[i] = mytstDatabuf[i-1]
end
t_tstData = torch.Tensor(recov_tstData, 1, #recov_tstData, 1)
t_tstData:view(tstSizes)

--train labels
local recov_trnLabel = torch.DoubleStorage(trnSizez[1])
for i=1, nTrnElem do
  recov_trnLabel[i] = mytrnLabelbuf[i-1]
end
t_trnLabel = torch.Tensor(recov_trnLabel, 1, #recov_trnLabel, 1)

--test labels
local recov_tstLabel = torch.DoubleStorage(tstSizez[1])
for i=1, nTrnElem do
  recov_tstLabel[i] = mytstLabelbuf[i-1]
end
t_tstLabel = torch.Tensor(recov_tstLabel, 1, #recov_tstLabel, 1)


mpi.barrier(comm)

--[[
  --------------------------------------------------
  --DISTRIBUTE THE MODEL----------------------------
  --------------------------------------------------
--]]



--[[
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 3, 3, 3))
model:add(nn.SpatialConvolutionMM(3, 4, 3, 3))
model:add(nn.SpatialMaxPooling(2,2,2,2))

local param = model:parameters()

print('model ' .. rank .. ' before bcast')
print(model)
print(param[1])

local t_param = model:getParameters()
local s_param = t_param:storage()
local count = #s_param

--create buffer for sending/receiving parameters
local buf = ffi.new("double[?]",count)

--node one will send its initial parameters to other nodes
if rank == 0 then  
  for i = 0, count-1 do
    buf[i] = s_param[i+1]
  end
end

mpi.bcast(buf, count, mpi.double, 0, comm)

if rank ~= 0 then  
  for i = 0, count-1 do
    s_param[i+1] = buf[i]
  end

  t_param = torch.Tensor(s_param, 1, #s_param, 1)
  --print(t_param)

  --fill model parameters with recieved parameters
  start = 1
  for i = 1, #param do
    nElem = param[i]:nElement() 
    stop = start + nElem - 1 
    t_elems = t_param[{{start, stop}}]
    t_elems = t_elems:viewAs(param[i])
 
    param[i] = t_elems
  
    start = start + nElem 
  end
end
print('model '..rank)
print(param[1])

--[[
  --------------------------------------------------
  --TRAINING----------------------------------------
  --------------------------------------------------
--]]
--]]

mpi.finalize()
