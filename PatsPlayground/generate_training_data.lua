  require 'torch'
  --just generating some random train/test data (later we'll load actual data)
  trainData = {
    data = torch.randn(nTrain, channels, height, width),
    labels = torch.Tensor(nTrain),
    size = function() return #labels end
  }
  i=0
  trainData.labels:apply(function(x) i=i%10+1 return i end) 

