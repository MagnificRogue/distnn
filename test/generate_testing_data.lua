require 'torch'



--some input parameters you can tweak easily
height = 16
width = 16
channels = 1
nTrain = 8000
nTest = 8000



testData = {
    data = torch.randn(torch.LongStorage{nTest, channels, height, width}),
    labels = torch.Tensor(nTest),
    size = function() return #labels end
  }
  i=0
  testData.labels:apply(function(x) i=i%10+1 return i end) 


