require('cutorch')
require('cunn')
require('dcnn')

local m = dcnn.ReLU()
m:cuda()

local x = torch.CudaTensor(4,4)
for i=1,4 do
	for j=1,2 do
        x[i][j]=-1
	end
end

for i=1,4 do
	for j=3,4 do
        x[i][j]=1
	end
end

print('==> Test forward ...' )
print(x)

print('==> Forward ...')
local y = m:forward(x)
print(y)


r = y:clone()
r[2][1] = 1
r[3][2] = 1
r[2][3] = -1
r[3][4] = -1
print('==> Test reconstruct ...' )
print(r)

print('==> Reconstruct ...' )
local z = m:reconstruct(x,r)
print(z)
