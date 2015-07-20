local torch = require("torch")
local ffi = require("ffi")
local mpi = require("mpi")

Input  = {}

Input.__index = Input

function Input.new( data, labels)
  local self = setmetatable({}, Input)

  assert(getmetatable(data) == getmetatable(torch.Tensor))
  assert(getmetatable(labels) == getmetatable(torch.Tensor))
  assert(getmetatable(input_count) == nil)

  self.data = data:type('torch.DoubleTensor')
  self.labels = labels:type('torch.DoubleTensor')

  --[[
      self.distributed_data is an array with the chunked data having been distributed
      and having each index as the appropriate rank for which ever node the chunk is on.

      Meaning, distributed_data[1] is the chunk that node 1 has

  --]]

  self.distributed_data = {}
  self.distributed_labels = {}


  self:distribute()




  return self
end

function Input.distribute(self) 
    mpi_info = {}
    mpi_info.comm = mpi.comm_world
    mpi_info.rank = mpi.comm_world:rank()
    mpi_info.size = mpi.comm_world:size()

    local tag = 15

    for key, item in ipairs(self.data:chunk( mpi_info.size , 4)) do
      local key = key - 1

      local stor = item:storage()
      local count = stor:size()

      local buff = ffi.new("double[?]",count)

      for i=1, count do
        buff[i-1] = stor[i]
      end


      if mpi_info.rank == 0 then
        if key > 0 then
          mpi.send(buff, count, mpi.double, key, tag, mpi_info.comm)
        end
      else if mpi_info.rank == key then
        mpi.recv(buff, count, mpi.double, 0, tag, mpi_info.comm)

        local recov_stor = torch.DoubleStorage(count)
        for i=1, count do
          recov_stor[i] = buff[i-1]
        end

        self.distributed_data[key] = torch.Tensor(recov_stor, item:size()[1], item:size()[2], item:size()[3], item:size()[4])

      end 
      end
    end

    mpi.barrier(mpi_info.comm)
  

    for key, item in ipairs(self.labels:chunk(mpi_info.size, 1)) do
      local key = key - 1

      local stor = item:storage()
      local count = stor:size()

      local buff = ffi.new("double[?]",count)

      for i=1, count do
        buff[i-1] = stor[i]
      end

      if mpi_info.rank == 0 then
        if key ~= 0 then
          mpi.send(buff, count, mpi.double, key, tag, mpi_info.comm)
        end
      else if mpi_info.rank == key then
        mpi.recv(buff, count, mpi.double, 0, tag, mpi_info.comm)

        local recov_stor = torch.DoubleStorage(count)
        
        for i=1, count do
          recov_stor[i] = buff[i-1]
        end

        self.distributed_labels[key] = torch.Tensor(recov_stor, item:size()[1])

      end
      end
    end    

  
  mpi.barrier(mpi_info.comm)

end

return Input
