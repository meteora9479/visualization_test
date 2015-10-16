local DualSpatialMaxUnpooling, parent = torch.class('unn.DualSpatialMaxUnpooling', 'nn.Module')

function DualSpatialMaxUnpooling:__init()
	parent.__init(self)
end

function DualSpatialMaxUnpooling:setDualModule(dualModule)
	self.dualModule = dualModule
	dualModule.dualModule = self
end

function DualSpatialMaxUnpooling:resizeOutput(dualInput)
	self.output:resizeAs(dualInput)
end

function DualSpatialMaxUnpooling:updateOutput(input)
	input.unn.DualSpatialMaxUnpooling_updateOutput(self.dualModule.indices, input, self.output)
	return self.output
end

function DualSpatialMaxUnpooling:updateGradInput(input, gradOutput)
	input.unn.DualSpatialMaxUnpooling_updateGradInput(self.dualModule.indices, input, gradOutput, self.gradInput)
	return self.gradInput
end

function DualSpatialMaxUnpooling:empty()
	self.gradInput:resize()
	self.gradInput:storage():resize(0)
	self.output:resize()
	self.output:storage():resize(0)
end

function DualSpatialMaxUnpooling:__tostring__()
	local s =  string.format('%s', torch.type(self))
	return s
end
