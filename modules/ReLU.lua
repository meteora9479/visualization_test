local ReLU, Parent = torch.class('dcnn.ReLU', 'dcnn.Threshold')

function ReLU:__init(p)
	Parent.__init(self,0,0,p)
end
