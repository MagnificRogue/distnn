require("torch")
require ("mpi")
require ("ffi")
require ("nn") 

[[-- Just a test for moving parameters between models of the same architecture.
     We create two really simple identical models, which get initialized with 
     different starting weights and biases. We then use getParameters() to extract
     the first model's weights and biases (in the form of a flat 1-D tensor).
     For funsies, we convert that tensor to a storage (we'll need to do this when we 
     start sending these via MPI) and then convert it back to a flat tensor again
     (again, this is what we'd do on the receiving end when we use MPI).
     Next, we loop through the second model's table of parameters and fill them
     with the corresponding parameters from the first model. This requires some
     indexing to extract the right subsets of the flat tensor
     and some reshaping to make the extracted subtensor fit the correct dimentions
     of the model's original parameters.
--]]

model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1, 3, 3, 3))
model:add(nn.SpatialConvolutionMM(3, 4, 3, 3))
model:add(nn.SpatialMaxPooling(2,2,2,2))


model2 = nn.Sequential()
model2:add(nn.SpatialConvolutionMM(1, 3, 3, 3))
model2:add(nn.SpatialConvolutionMM(3, 4, 3, 3))
model2:add(nn.SpatialMaxPooling(2,2,2,2))


tb_param1 = model:parameters()
tb_param2 = model2:parameters()

t_param1 = model:getParameters()
t_param2 = model:getParameters()

s_param1 = t_param1:storage()

print('model1')
print(model)
print(tb_param1[1])

print('model2')
print(model2)
print(tb_param2[1])


t_param1 = torch.Tensor(s_param1, 1, #s_param1, 1)
--print(t_param1)

start = 1
for i = 1, #tb_param2 do
  size = #tb_param2[i]
  nElem = tb_param2[i]:nElement() 
  stop = start + nElem - 1
  print('nElem ' .. nElem)
  print('start ' .. start .. '  stop ' .. stop)
  t_elems = t_param1[{{start, stop}}]
  t_elems = t_elems:viewAs(tb_param2[i])
  print('t_elems at ' .. i)
  print(t_elems)
 
  tb_param2[i] = t_elems
  
  start = start + nElem
end

print('model1')
print(tb_param1[1])
print('model2')
print(tb_param2[1])

--local comm = mpi.comm_world
--local rank = comm:rank()
--local size = comm:size()

--if rank == 0 then
  --local weight = model
  --local buff = ffi.new("double[?]",count)




