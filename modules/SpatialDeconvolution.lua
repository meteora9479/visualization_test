require'cudnn'
local SpatialDeconvolution, parent = torch.class('cudnn.SpatialDeconvolution', 'cudnn.SpatialConvolution')


function SpatialDeconvolution:__init( convLayer, reconstruction_size, firstDeconv )
    assert( torch.typename(convLayer) == 'cudnn.SpatialConvolution', 'Input should be cudnn.SpatialConvolution()')
    local deconvInputPlane = 1
    if firstDeconv == true then 
        deconvInputPlane = convLayer.nOutputPlane
    end
    
    parent.__init(self, deconvInputPlane, convLayer.nInputPlane, convLayer.kW, convLayer.kH, convLayer.dW, 
                  convLayer.dH, convLayer.padW, convLayer.padH, convLayer.groups)
    self:reset() 
    self.reconstruction_size = reconstruction_size
    self.weight=convLayer.weight
    self.gradWeight=convLayer.gradWeight    
end



function SpatialDeconvolution:updateOutput(input)
    local deconv1_fm = torch.CudaTensor( self.nInputPlane, self.nOutputPlane, self.reconstruction_size, 
                                         self.reconstruction_size):zero():cuda()
    local deconv1 = cudnn.SpatialConvolution(1, 1, self.kW, self.kH, 1, 1, 
                                             math.floor(self.kW/2), math.floor(self.kH/2), self.group):cuda()
    
    local conv_scat_fm = torch.zeros(self.nInputPlane, self.reconstruction_size, self.reconstruction_size):cuda()
    local stride_size = self.dH
    local padding_size = (self.reconstruction_size - (input:size(2) * stride_size))/2
    
    local n=input:size(2)
    local x=stride_size

    if self.reconstruction_size ~= input:size(2) then
        -- Scatter     contributed by TingFan
        local idx=torch.LongTensor(n*n,1):cuda()
        local counter=1;
        for i=1,n*x,x do
            for j=1,n*x,x do     
                idx[counter]=(i-1+padding_size)*(n*x+padding_size*2) + j + padding_size
                counter=counter+1;
            end
        end

        local total_size = (n*x+padding_size*2)*(n*x+padding_size*2)
        for i=1,self.nInputPlane do
            local m=torch.zeros(n*x+padding_size*2,n*x+padding_size*2):cuda()
            m:view(total_size,1):scatter(1,idx,input[i]:view(n*n,1))
            conv_scat_fm[i] = m
        end    
    end  
        
    -- Deconv
    for i=1, self.nInputPlane do
        for j=1, self.nOutputPlane do
            local fm = conv_scat_fm[i]

            deconv1.weight = self.weight[{ {i}, {j}, {}, {} }]:transpose(3, 4):contiguous()
            local deconv_result = deconv1:forward(fm:view(1, self.reconstruction_size, self.reconstruction_size)):cuda()

            -- BGR to RGB
            if self.nOutputPlane==3 then
                deconv1_fm[{ {i}, {3-(j-1)}, {}, {} }] = deconv_result
            else
                deconv1_fm[{ {i}, {j}, {}, {} }] = deconv_result
            end
            
        end
    end

    cutorch.synchronize()    
    return deconv1_fm
end 

