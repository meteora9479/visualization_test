require 'cudnn'
require 'inn'
require 'image'
require 'dcnn'
gfx = require 'gfx.js'

-- Loads the mapping from net outputs to human readable labels
function load_synset()
  local file = io.open 'synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  return list
end

-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')*255
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  return im4 - image.scale(img_mean, 224, 224, 'bilinear')
end

-- Setting up networks 
print '==> Loading network'

--net = torch.load('/home/yusheng/Workspace/DeepLearning/models/zeilerNet/zeilerNet.net')
net = torch.load('/usr/local/data/zeilerNet/zeilerNet.net')
net:cuda()
-- as we want to classify, let's disable dropouts by enabling evaluation mode
net:evaluate()

print '==> Loading synsets'
synset_words = load_synset()

print '==> Loading image and imagenet mean'
image_name = 'Goldfish3.jpg'
--image_name = 'lena.jpg'
--image_name='people2.jpg'
img_mean_name = 'ilsvrc_2012_mean.t7'

im = image.load(image_name)
img_mean = torch.load(img_mean_name).img_mean:transpose(3,1)

-- Have to resize and convert from RGB to BGR and subtract mean
print '==> Preprocessing'
I = preprocess(im, img_mean)

-- Replace pooling by dual pooling
unpooling_layers = dcnn:ReplaceDualPoolingModule(net:get(1))

_,classes = net:forward(I:cuda()):view(-1):float():sort(true)

for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end


-- set deconvNet
local deconvNet = nn.Sequential()

local conv5_fm = net:get(1):get(13).output

require'testing_deconvLayer'

-- deconvNet:add(Test_SpatialDeconvolution( net:get(1):get(13), net:get(1):get(11).output:size(2), 27 ))
-- deconvNet:add(cudnn.ReLU(true))
-- deconvNet:add(Test_SpatialDeconvolution( net:get(1):get(11), net:get(1):get(9).output:size(2), true ))
-- deconvNet:add(cudnn.ReLU(true))
-- deconvNet:add(Test_SpatialDeconvolution( net:get(1):get(9), net:get(1):get(8).output:size(2), true ))
-- deconvNet:add(unpooling_layers[2])
-- deconvNet:add(cudnn.ReLU(true))
-- deconvNet:add(Test_SpatialDeconvolution( net:get(1):get(5), net:get(1):get(4).output:size(2), true ))
-- deconvNet:add(unpooling_layers[1])
-- deconvNet:add(cudnn.ReLU(true))
-- deconvNet:add(Test_SpatialDeconvolution( net:get(1):get(1), I:size(2), true ))

deconv1 = Test_SpatialDeconvolution( net:get(1):get(1), I:size(2), 27 )
conv1 = net:get(1):get(1)

-- vis_c1 = deconv1:forward( net:get(1):get(1).output )
-- print(vis_c1:size() )
-- gfx.image(vis_c1)


-- error_tensor = 0
-- last_output = 0
-- local last_time = 0
-- layer_n = 1


for i=1,20 do
    deconv_timer = torch.Timer()
    --vis_c5 = deconvNet:forward(conv5_fm)
    vis_c1 = deconv1:forward( net:get(1):get(1).output )
    --conv1_fm = conv1:forward(I:cuda()) 
    
    print('==> Time elapsed: ' .. deconv_timer:time().real .. ' seconds')
    
    if i==1 then
        last_output = vis_c1
        -- last_output = conv1.output 
        -- last_output = scat_fm
        -- last_output = weight
    end
    
    error_tensor = last_output - vis_c1
    -- error_tensor = last_output - conv1.output
    -- error_tensor = last_output - scat_fm
    -- error_tensor = last_output - weight
    local test_error = 0
    if type(error_tensor) ~= 'number' then 
        for j=1, error_tensor:view(-1):size(1) do
            test_error = test_error + error_tensor:view(-1)[j]
        end    
        
        print( error_tensor:type() )
    else
        test_error = error_tensor
        print( type(error_tensor) )
    end
    
    print(test_error)
    --gfx.image(image.toDisplayTensor( {input=net:get(1):get(13).output, padding=2, nrow=16} ) )
    --gfx.image(vis_c5)
    --gfx.image(vis_c1)
end


