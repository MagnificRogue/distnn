require("torch")
require ("nn") 
local mpi = require ("mpi")
local ffi =require ("ffi")


--[[ model:parameters() returns a table of model's parameters.
     Ex: param = 
          {
            1: DoubleTensor - size 3x9  (weights of conv layer 1)
            2: DoubleTensor - size 3    (biases of conv layer 1)
            3: DoubleTensor - size 4x27 (weights of conv layer 2)
            4: DoubleTensor - size 4    (biases of conv layer 2)
          }
     Any changes to param are made to the model itself.
--]]

local comm = mpi.comm_world
local rank = comm:rank()
local size = comm:size()

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






