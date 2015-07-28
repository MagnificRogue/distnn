require 'torch'

testData = {
    data = torch.randn(nTest, channels, height, width),
    labels = torch.Tensor(nTest),
    size = function() return #labels end
  }
  i=0
  testData.labels:apply(function(x) i=i%10+1 return i end) 


