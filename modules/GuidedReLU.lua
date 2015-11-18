local GuidedReLU, Parent = torch.class('dcnn.GuidedReLU', 'dcnn.Threshold')

function GuidedReLU:__init( guided_layer )
    Parent.__init(self,0,0,false)
    if guided_layer ~= nil then
        self.deconv_mode = true
        self.guided_layer = guided_layer
    else
       self.deconv_mode = false 
    end
    
    self.output = self.guided_layer.output
end

function GuidedReLU:updateOutput(input)
    if self.deconv_mode == true  then
        self:validateParameters()
        
        assert( self.guided_layer.output:dim()==input:dim() , 'Reconstructed dimension must be same as input dimension !' )
        for i=1, input:dim() do
            assert( self.guided_layer.output:size(i)==input:size(i) , 'Reconstructed size must be same as input size !' )
        end
            
        input.dcnn.Threshold_updateRecOutput( self, self.guided_layer.output, input )
        return self.recOutput
    else
       self:validateParameters()
       input.nn.Threshold_updateOutput(self, input)
       return self.output 
    end
end