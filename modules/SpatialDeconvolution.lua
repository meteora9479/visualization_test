require'cudnn'
local SpatialDeconvolution, parent = torch.class('cudnn.SpatialDeconvolution', 'cudnn.SpatialConvolution')


function SpatialDeconvolution:__init( convLayer, reconstruction_size, neuron_num, normal_deconv )
    assert( torch.typename(convLayer) == 'cudnn.SpatialConvolution', 'Input should be cudnn.SpatialConvolution()')
    parent.__init(self, convLayer.nOutputPlane, convLayer.nInputPlane, convLayer.kW, convLayer.kH, convLayer.dW, 
                  convLayer.dH, convLayer.padW, convLayer.padH, convLayer.groups)
    self:reset() 
    
    self.normal_deconv = normal_deconv or false
    self.neuron_num = neuron_num or 0
    self.reconstruction_size = reconstruction_size
    self.weight=convLayer.weight
    self.gradWeight=convLayer.gradWeight    
end



function SpatialDeconvolution:updateOutput(input)
    --print(self.normalDeconv)
    local deconv_output = self.nInputPlane
    local total_deconv = self.nInputPlane
    if self.neuron_num ~= 0 then
        total_deconv = 1
        deconv_output = 1
    end    
    
    if normal_deconv == true then
        deconv_output = 1
    end
        
    local deconv_fm = torch.CudaTensor( deconv_output, self.nOutputPlane, self.reconstruction_size, 
                                         self.reconstruction_size):zero():cuda()
    local deconv = cudnn.SpatialConvolution(1, 1, self.kW, self.kH, 1, 1, 
                                            math.floor(self.kW/2), math.floor(self.kH/2), self.group):cuda()
    
    local conv_scat_fm = torch.zeros(total_deconv, self.reconstruction_size, self.reconstruction_size):cuda()
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
        for i=1,total_deconv do
            local m=torch.zeros(n*x+padding_size*2,n*x+padding_size*2):cuda()
            m:view(total_size,1):scatter(1,idx,input[i]:view(n*n,1))
            conv_scat_fm[i] = m
        end    
    end  
        
    -- Deconv
    if self.normal_deconv == false then
        for i=1, total_deconv do
            for j=1, self.nOutputPlane do
                local weight_index = i
                if self.neuron_num ~= 0 then
                    weight_index = self.neuron_num
                end

                local fm = conv_scat_fm[i]
                deconv.weight = self.weight[{ {weight_index}, {j}, {}, {} }]:transpose(3, 4):contiguous()
                local deconv_result = deconv:forward(fm:view(1, self.reconstruction_size, self.reconstruction_size)):cuda()
                -- BGR to RGB
                if self.nOutputPlane==3 then
                    deconv_fm[{ {i}, {3-(j-1)}, {}, {} }] = deconv_result
                else
                    deconv_fm[{ {i}, {j}, {}, {} }] = deconv_result
                end            
            end
        end
    else
        for j=1, self.nOutputPlane do 
            deconv.weight = self.weight[{ {}, {j}, {}, {} }]:transpose(3, 4):contiguous()
            local deconv_result = deconv:forward(fm:view(1, self.reconstruction_size, self.reconstruction_size)):cuda()
            -- BGR to RGB
            if self.nOutputPlane==3 then
                deconv_fm[{ {1}, {3-(j-1)}, {}, {} }] = deconv_result
            else
                deconv_fm[{ {1}, {j}, {}, {} }] = deconv_result
            end
        end
    end   
     
    cutorch.synchronize()    
    return deconv_fm
end 

