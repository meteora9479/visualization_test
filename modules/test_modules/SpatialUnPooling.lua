require 'cudnn'
local SpatialUnPooling, parent = torch.class('cudnn.SpatialUnPooling', 'cudnn.SpatialMaxPooling')


function SpatialUnPooling:__init( poolLayer, conv_fm )
    assert( torch.typename(poolLayer) == 'cudnn.SpatialMaxPooling', 'Input should be cudnn.SpatialMaxPooling()')
    parent.__init(self, poolLayer.kW, poolLayer.kH, poolLayer.dW, poolLayer.dH, poolLayer.padW, poolLayer.padH )
    self:reset() 
    self.reconstruction_size = conv_fm:size(2)
    self.switches = self:creatSwitchesTable(conv_fm:cuda())
    cutorch.synchronize()
end

function SpatialUnPooling:creatSwitchesTable(conv_fm)
    local stride_size = self.dW
    local kernel_size = self.kW
    local switches={}
    
    conv_fm = nn.SpatialZeroPadding(0, 0, 0, 0 ):forward(conv_fm:cuda())
    for i=1,conv_fm:size(1) do
        local x=1
        local counter =1
        switches[i]={}
        while x<=conv_fm:size(2) do
            local y=1
            while y<=conv_fm:size(3) do
                local x_ub = x+kernel_size-1
                local y_ub = y+kernel_size-1
                if x_ub > conv_fm:size(2) then
                    x_ub = conv_fm:size(2)
                end

                if y_ub > conv_fm:size(3) then
                    y_ub = conv_fm:size(3)
                end          

                local max_kernel = conv_fm[{{i},{x,x_ub},{y,y_ub}}]            
                max_col,idx_col=torch.max(max_kernel[1],2)
                max_val,idx_row=torch.max(max_col,1)
                switches[i][counter] = { max_row=x-1+idx_row[1][1], 
                                           max_col=y-1+idx_col[idx_row[1][1]][1] }
                counter = counter + 1    
                y=y+stride_size
            end

            x=x+stride_size
        end
    end   
    
    return switches
end

function SpatialUnPooling:updateOutput( pool_fm )
    local unpool_fm = torch.CudaTensor( pool_fm:size(1), self.reconstruction_size , self.reconstruction_size ):zero():cuda()
    
    for i=1,pool_fm:size(1) do
        local row=1
        local col=1
        for j in pairs(self.switches[i]) do
            unpool_fm[i][self.switches[i][j].max_row][self.switches[i][j].max_col] = pool_fm[i][row][col]
            if col+1<=pool_fm:size(2) then
                col = col +1
            else
                col = 1
                row = row + 1
            end
        end
    end  
    
    return unpool_fm
end 
