local torch = require 'torch'
mpi = require 'mpi'

local distnn = {}

distnn.hellomodule = require 'distnn.hellomodule'
distnn.Input = require 'distnn.Input'

return distnn
