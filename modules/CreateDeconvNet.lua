function dcnn:CreateDeconvNet( net, unpooling_layers, input_img, layer_num, neuron_index, normal_deconv  )
    neuron_index = neuron_index or 0
    if neuron_index == true then
        neuron_index = 0 
        normal_deconv = true
    end
    
    if neuron_index == false then
        neuron_index = 0 
        normal_deconv = false
    end 
    
    local deconvNet = nn.Sequential()
    local first_deconv = true
    for i=layer_num,1,-1 do
        if torch.typename( net:get(i)) == 'cudnn.SpatialConvolution' then
            local reconstructed_size = 0
            if i==1 then
                reconstructed_size = input_img:size(2)
            else    
                reconstructed_size = net:get(i-1).output:size(2)
            end
                
            if first_deconv == true then
                --print( net:get(i) )
                if neuron_index ~= 0 then 
                    deconvNet:add( cudnn.SpatialDeconvolution( net:get(i), reconstructed_size, neuron_index ))
                    --print(neuron_index)
                else
                    deconvNet:add( cudnn.SpatialDeconvolution( net:get(i), reconstructed_size, normal_deconv ))
                end
                
                first_deconv = false
            else
                deconvNet:add( cudnn.SpatialDeconvolution( net:get(i), reconstructed_size, true ))
            end
            
        elseif torch.typename( net:get(i)) == 'dcnn.DualSpatialMaxPooling' then
            local unpooling_idx = 1
            for j=i,1,-1 do  
                if j-1 > 1 and torch.typename( net:get(j-1)) == 'dcnn.DualSpatialMaxPooling' then
                   unpooling_idx = unpooling_idx + 1
                end
            end
            
            deconvNet:add( unpooling_layers[unpooling_idx])
            
        elseif torch.typename( net:get(i)) == 'cudnn.ReLU' then
            deconvNet:add( cudnn.ReLU(true) )
            
        else
            print( torch.typename( net:get(i))..' This type of layer is not supported !')
        end
        
        --print('Layer '..i..' is complete ')
    end    
        
    --deconvNet:cuda()
    return deconvNet
end