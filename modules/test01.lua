require('cutorch')
require('cunn')
require('unn')

torch.setdefaulttensortype('torch.FloatTensor')

local pool1 = unn.DualSpatialMaxPooling(2,2,2,2)
local pool2 = unn.DualSpatialMaxUnpooling()
pool1:cuda()
pool2:cuda()
-- must set dual moule after cuda
pool2:setDualModule(pool1)

local len = 8
local x = torch.CudaTensor(1,1,len,len):fill(1)
for i=1,len,2 do
	for j=1,len,2 do
		x[1][1][i][j] = 2 + j
	end
end
print(x)
local y1 = pool1:forward(x)
print(y1)
local y2 = pool2:forward(y1)
print(y2)
local z2 = pool2:backward(y1,y2)
print(z2)
local z1 = pool1:backward(x,z2)
print(z1)
