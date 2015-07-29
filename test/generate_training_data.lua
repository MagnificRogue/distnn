  
--some input parameters you can tweak easily
height = 16
width = 16
channels = 1
nTrain = 8000
nTest = 8000


  
  
  require 'torch'
  --just generating some random train/test data (later we'll load actual data)
  trainData = {
    data = torch.randn(torch.LongStorage{nTrain, channels, height, width}),
    labels = torch.Tensor(nTrain),
    size = function() return #labels end
  }
  i=0
  trainData.labels:apply(function(x) i=i%10+1 return i end) 

