function dcnn:ReplaceDualPoolingModule( nn_sequential ) 
    local unpooling_layers = {}
    assert( torch.typename(nn_sequential) == 'nn.Sequential', 'Input should be nn.Sequential()')
    amount_of_poolingLayers = 0
    for i=1, nn_sequential:size() do
        local layer_name = torch.typename(nn_sequential:get(i))
        if layer_name == 'cudnn.SpatialMaxPooling' then
            amount_of_poolingLayers = amount_of_poolingLayers + 1
            local dsmp = dcnn.DualSpatialMaxPooling( nn_sequential:get(i).kW, nn_sequential:get(i).kH, nn_sequential:get(i).dW, 
                                        nn_sequential:get(i).dH, nn_sequential:get(i).padW, nn_sequential:get(i).padH)
            local dsmup = dcnn.DualSpatialMaxUnpooling()
            dsmp:cuda()
            dsmup:cuda()
            dsmup:setDualModule(dsmp)
            nn_sequential.modules[i] = dsmp
            unpooling_layers[amount_of_poolingLayers]=dsmup
        end
    end    
    
    return unpooling_layers
end    
