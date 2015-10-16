local SpatialMaxUnpooling, parent = torch.class('nn.SpatialMaxUnpooling', 'nn.Module')

function SpatialMaxUnpooling:__init()
	parent.__init(self)
end

function SpatialMaxUnpooling:setDualModule(dualModule)
	self.dualModule = dualModule
	dualModule.dualModule = self
end

function SpatialMaxUnpooling:updateOutput(input)
	input.unn.DualSpatialMaxUnpooling_updateOutput(self.dualModule.indices, input, self.output)
	return self.output
end


function SpatialMaxUnpooling:empty()
	self.gradInput:resize()
	self.gradInput:storage():resize(0)
	self.output:resize()
	self.output:storage():resize(0)
end

function SpatialMaxUnpooling:__tostring__()
	local s =  string.format('%s', torch.type(self))
	return s
end



