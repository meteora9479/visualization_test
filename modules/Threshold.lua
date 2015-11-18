local Threshold, parent = torch.class('dcnn.Threshold','nn.Threshold')

function Threshold:__init(th,v,ip)
	parent.__init(self,th,v,ip)

	self.recOutput = torch.Tensor()
    self:cuda()
end

function Threshold:updateRecOutput(input, recInput)
	self:validateParameters()
	input.dcnn.Threshold_updateRecOutput(self, input, recInput)
	return self.recOutput
end

function Threshold:reconstruct(input, recInput)
	return self:updateRecOutput(input, recInput)
end
