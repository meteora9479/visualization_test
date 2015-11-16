local Test_SpatialDeconvolution, parent = torch.class('Test_SpatialDeconvolution', 'cudnn.SpatialConvolution')

function Test_SpatialDeconvolution:__init( convLayer, reconstruction_size, neuron_num )
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
    
    if self.neuron_num == false then
        self.neuron_num = 0 
        self.normal_deconv = false
    end 
    
    
    self.reconstruction_size = reconstruction_size
    self.weight=convLayer.weight:clone()
        
    flip = function(m,d) return m:index(d,torch.range(m:size(d),1,-1):long())end
    self.weight = flip(flip(self.weight,4),3)
    self.gradWeight=convLayer.gradWeight    
    
    self.deconv = {}
    for i=1, convLayer.nOutputPlane do
        self.deconv[i] = cudnn.SpatialConvolution(1, convLayer.nInputPlane, self.kW, self.kH, 1, 1, 
                                             math.floor(self.kW/2), math.floor(self.kH/2) ):cuda()

        self.deconv[i].weight = self.weight[{{i}, {}, {}, {} }]:transpose(1,2):contiguous():cuda()
    end

    self.deconv_normal = cudnn.SpatialConvolution( self.nInputPlane, self.nOutputPlane, self.kW, self.kH, 1, 1, 
                                                        math.floor(self.kW/2), math.floor(self.kH/2)):cuda()         
    self.deconv_normal.weight = self.weight:transpose(1, 2):contiguous():cuda()    
        
    print( convLayer:__tostring__() )
    print('deconv layer has been built !')
end

function Test_SpatialDeconvolution:updateOutput(input)
    local deconv_output = self.nInputPlane
    local total_deconv = self.nInputPlane
    if torch.type(self.neuron_num)~='torch.IntTensor' and self.neuron_num ~= 0 then
        total_deconv = 1
        deconv_output = 1
    end    
    
    if torch.type(self.neuron_num) == 'torch.IntTensor' then
        total_deconv = self.neuron_num:size(1)
        deconv_output = 1
    end
    
    if normal_deconv == true then
        deconv_output = 1
    end
            
    local deconv_fm = torch.CudaTensor( deconv_output, self.nOutputPlane, self.reconstruction_size, 
                                         self.reconstruction_size):cuda():zero()
    
    local conv_scat_fm = torch.zeros(total_deconv, self.reconstruction_size, self.reconstruction_size):cuda()
    local stride_size = self.dH
    local padding_size = (self.reconstruction_size - (input:size(2) * stride_size))/2
    local increased_size = self.reconstruction_size - (input:size(2) * stride_size)
    
    local n=input:size(2)
    local x=stride_size
    input:cuda()
   
    --timer = torch.Timer()
    if self.reconstruction_size ~= input:size(2) then
        -- Scatter     contributed by TingFan
        local idx=torch.LongTensor(n*n,1):cuda()
        local counter=1;
        for i=x,n*x,x do
            for j=x,n*x,x do
                idx[counter]=(i+math.floor(padding_size)) *(n * x + increased_size) + j + math.floor(padding_size)
                counter=counter+1;
            end
        end

        local total_size = (n*x+increased_size)*(n*x+increased_size)
        for i=1,total_deconv do
            local fm_index = i
            
            if torch.type(self.neuron_num) ~= 'torch.IntTensor' then
                if self.neuron_num ~= 0 then
                     fm_index = self.neuron_num
                end
            else
                fm_index = self.neuron_num[i]
            end
                
            local m=torch.zeros(n*x+increased_size,n*x+increased_size):cuda()
            local output = input[fm_index]:view(n*n,1)
            m:view(total_size,1):scatter(1,idx,output)
            conv_scat_fm[i] = m
        end      
    else
        if torch.type(self.neuron_num) ~= 'torch.IntTensor' and self.neuron_num ~= 0 then
            conv_scat_fm[{{1},{},{}}] = input[{{self.neuron_num},{},{}}]:clone()
        elseif torch.type(self.nuuron_num) == 'torch.IntTensor' then
            for i=1,self.nuuron_num:size(1) do
                conv_scat_fm[i] = input[{{self.neuron_num[i]},{},{}}]
            end
        else
            conv_scat_fm = input
        end
    end
        
    --print('==> Scatter Time elapsed: ' .. timer:time().real .. ' seconds')
    --timer2 = torch.Timer()
    -- local test_return = nil
    
    -- Deconv
    if self.normal_deconv == false then
        if torch.type(self.neuron_num) ~= 'torch.IntTensor' then -- single_neuron
            for i=1, total_deconv do
                local weight_index = i
                if self.neuron_num ~= 0 then
                    weight_index = self.neuron_num
                end

                local fm = conv_scat_fm[i]                    
                local deconv_result = self.deconv[weight_index]:forward(fm:view(1, self.reconstruction_size, 
                                                                        self.reconstruction_size):cuda())
                deconv_fm[i] = deconv_result:contiguous():cuda()
--                 --test deconv_result
--                 if j==3 then
--                     -- test_return = deconv.weight[1][1]:clone()
--                     test_return = deconv_result:clone()
--                 end

--                 local error_tensor = deconv_result - deconv_fm
--                 local test_error = 0
--                 for j=1, error_tensor:view(-1):size(1) do
--                     test_error = test_error + error_tensor:view(-1)[j]
--                 end

--                 print( 'error_fm:  ' .. test_error )
                                        
                --print( 'single_neuron' )

            end      
        else
            for i=1, total_deconv do 
                local weight_index = self.neuron_num[i]
                local fm = conv_scat_fm[i]
                local deconv_result = self.deconv[weight_index]:forward(fm:view(1, self.reconstruction_size,
                                                                        self.reconstruction_size)):cuda()
                deconv_fm = deconv_fm + deconv_result                     
            end
            
            --print( 'multi_neuron' )
        end
    else                  
        deconv_fm = self.deconv_normal:forward(conv_scat_fm):cuda()        
        --print( 'normal_deconv' )
    end   
    
    --print('==> Deconv Time elapsed: ' .. timer2:time().real .. ' seconds')
        
    cutorch.synchronize()
    if deconv_fm:dim() == 4 and deconv_output==1 then
        print(deconv_fm:size())
        -- print('dim == 4' )
        if self.nOutputPlane==3 then
          local temp = deconv_fm:clone()
          deconv_fm[{{},1,{},{}}] = temp[{{},3,{},{}}]
          deconv_fm[{{},3,{},{}}] = temp[{{},1,{},{}}]   
        end        
        
        self.output = deconv_fm[1]
        return deconv_fm[1]
    end    
    
    -- BGR to RGB
    if self.nOutputPlane==3 then
      local temp = deconv_fm:clone()
      deconv_fm[{1,{},{}}] = temp[{3,{},{}}]
      deconv_fm[{3,{},{}}] = temp[{1,{},{}}]   
    end     
    
    self.output = deconv_fm
    return deconv_fm
end 