--[[
      This assumes that torch has been required prior and that MPI is running
      and that ffi has been called. IF this is not the case, it can be via the following
      includes:

        local torch = require 'torch'
        local mpi = require 'mpi'
        local ffi = require 'ffi'


-]]



mpi.send_tensor = function(tensor, receiver_rank, tag, com_world)


  local buffer_size = ffi.new("int[?]", tensor:size():size())
          
  for i=1,tensor:size():size() do  
    buffer_size[i-1] = tensor:size()[i]
  end


  mpi.send(buffer_size, tensor:size():size(), mpi.int, receiver_rank, tag, com_world)


 -------------------------------------------------------------- Second, SEND the buffer


 local stor = tensor:storage()
 local count = stor:size()

 local tensor_buffer = ffi.new("double[?]",count)
 for i=1, count do
   tensor_buffer[i-1] = stor[i]
 end
        
 mpi.send(tensor_buffer, count, mpi.double, receiver_rank, tag, com_world)

end




mpi.receive_tensor = function(tensor_dimensionality, sender_rank, tag, comm_world)


  ----First, recover the size
  local recovered_size = ffi.new("int[?]",tensor_dimensionality)
  local recovered_size_storage = torch.LongStorage(tensor_dimensionality)

  mpi.recv(recovered_size, tensor_dimensionality, mpi.int, 0, tag, comm_world)

  for i=1,tensor_dimensionality do
    recovered_size_storage[i] = recovered_size[i-1]
  end

  recovered_size = recovered_size_storage


  -------Now that we have recovered the size of our chunk, lets get the size of the next chunk we're getting
  -------and recover that chunk too!

  local count = 1

  for i=1,tensor_dimensionality do
    count = count * recovered_size[i]
  end
  
  local recovered_chunk = ffi.new("double[?]", count)
  local recovered_chunk_storage = torch.DoubleStorage(count)

  mpi.recv(recovered_chunk, count, mpi.double, 0, tag, comm_world)

  for i=1,count do
    recovered_chunk_storage[i] = recovered_chunk[i-1]
  end

  recovered_chunk = recovered_chunk_storage


  return torch.Tensor(recovered_chunk,1,recovered_size)


end



