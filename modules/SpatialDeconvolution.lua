require 'cudnn'
require 'image'
local SpatialDeconvolution, parent = torch.class('cudnn.SpatialDeconvolution', 'cudnn.SpatialConvolution')


function SpatialDeconvolution:__init( convLayer, reconstruction_size, neuron_num, normal_deconv )
    assert( torch.typename(convLayer) == 'cudnn.SpatialConvolution', 'Input should be cudnn.SpatialConvolution()')
    parent.__init(self, convLayer.nOutputPlane, convLayer.nInputPlane, convLayer.kW, convLayer.kH, convLayer.dW, 
                  convLayer.dH, convLayer.padW, convLayer.padH, convLayer.groups)
    self:reset() 
    
    self.normal_deconv = normal_deconv or false
    self.neuron_num = neuron_num or 0
    if self.neuron_num == true then
       self.neuron_num = 0 
       self.normal_deconv = true
    end
    
    self.reconstruction_size = reconstruction_size
    self.weight=convLayer.weight:clone()
        
    flip = function(m,d) return m:index(d,torch.range(m:size(d),1,-1):long())end
    self.weight = flip(flip(self.weight,4),3)
    self.gradWeight=convLayer.gradWeight    
end



function SpatialDeconvolution:updateOutput(input)
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
    
    timer = torch.Timer()
    if self.reconstruction_size ~= input:size(2) then
        -- Scatter     contributed by TingFan
        local idx=torch.LongTensor(n*n,1):cuda()
        local counter=1;
        for i=x,n*x,x do
            for j=x,n*x,x do
                idx[counter]=(i+math.floor(padding_size))*(n*x+padding_size*2) + j + math.floor(padding_size)
                counter=counter+1;
            end
        end

        local total_size = (n*x+padding_size*2)*(n*x+padding_size*2)
        for i=1,total_deconv do
            local fm_index = i
            if self.neuron_num ~= 0 then
                 fm_index = self.neuron_num
            end
            
            local m=torch.zeros(n*x+padding_size*2,n*x+padding_size*2):cuda()
            local output = input[fm_index]:view(n*n,1)
            m:view(total_size,1):scatter(1,idx,output)
            conv_scat_fm[i] = m
        end      
    else
        if self.neuron_num ~= 0 then
            conv_scat_fm[{{1},{},{}}] = input[{{self.neuron_num},{},{}}]
        else
            conv_scat_fm = input:cuda()
        end
    end
        
    print('==> Scatter Time elapsed: ' .. timer:time().real .. ' seconds')
    timer2 = torch.Timer()
    -- Deconv
    if self.normal_deconv == false then
        for i=1, total_deconv do
            for j=1, self.nOutputPlane do
                local weight_index = i
                if self.neuron_num ~= 0 then
                    weight_index = self.neuron_num
                end

                local fm = conv_scat_fm[i]
                --deconv.weight = self.weight[{ {weight_index}, {j}, {}, {} }]:transpose(3, 4):contiguous()
                deconv.weight = self.weight[weight_index][j]
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
        local deconv_normal = cudnn.SpatialConvolution( self.nInputPlane, self.nOutputPlane, self.kW, self.kH, 1, 1, 
                                                        math.floor(self.kW/2), math.floor(self.kH/2), self.group):cuda()         
        deconv_normal.weight = torch.CudaTensor( self.nOutputPlane, self.nInputPlane, self.kW, self.kH )
        
        timer2 = torch.Timer()
        deconv_normal.weight = self.weight:transpose(1, 2):contiguous()          
        deconv_fm = deconv_normal:forward(conv_scat_fm):cuda()
        -- BGR to RGB
        if self.nOutputPlane==3 then
          local temp = deconv_fm:clone()
          deconv_fm[{1,{},{}}] = temp[{3,{},{}}]
          deconv_fm[{3,{},{}}] = temp[{1,{},{}}]   
        end
    end   
    
    print('==> Deconv Time elapsed: ' .. timer2:time().real .. ' seconds')
    cutorch.synchronize()    
    return deconv_fm
end 

