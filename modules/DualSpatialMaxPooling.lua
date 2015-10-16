local DualSpatialMaxPooling, parent = torch.class('dcnn.DualSpatialMaxPooling', 'nn.Module')

function DualSpatialMaxPooling:__init(kW, kH, dW, dH, padW, padH)
	parent.__init(self)

	dW = dW or kW
	dH = dH or kH

	self.kW = kW
	self.kH = kH
	self.dW = dW
	self.dH = dH

	self.padW = padW or 0
	self.padH = padH or 0

	self.indices = torch.Tensor()
end

function DualSpatialMaxPooling:updateOutput(input)
	if self.dualModule then
		self.dualModule:resizeOutput(input)
	end
	input.dcnn.DualSpatialMaxPooling_updateOutput(self, input)
	return self.output
end

function DualSpatialMaxPooling:updateGradInput(input, gradOutput)
	input.dcnn.DualSpatialMaxPooling_updateGradInput(self, input, gradOutput)
	return self.gradInput
end

function DualSpatialMaxPooling:empty()
	self.gradInput:resize()
	self.gradInput:storage():resize(0)
	self.output:resize()
	self.output:storage():resize(0)
	self.indices:resize()
	self.indices:storage():resize(0)
end

function DualSpatialMaxPooling:__tostring__()
	local s =  string.format('%s(%d,%d,%d,%d', torch.type(self),
		self.kW, self.kH, self.dW, self.dH)
	if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
		s = s .. ',' .. self.padW .. ','.. self.padH
	end
	s = s .. ')'

	return s
end
