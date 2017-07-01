require 'cudnn'
require 'nn'

local function c3d(batchSize)
   -- Create table describing C3D configuration
   local cfg = {64, 'P1', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'}
   -- convolutional layers
   local conv_layers = nn.Sequential()
   do
      local iChannels = 3;
      for i = 1, #cfg do
         if cfg[i] == 'P' then
            conv_layers:add(nn.VolumetricMaxPooling(2,2,2,2,2,2):ceil())
         elseif cfg[i] == 'P1' then
            conv_layers:add(nn.VolumetricMaxPooling(1,2,2,1,2,2):ceil())
         else
            local oChannels = cfg[i];
            conv_layers:add(nn.VolumetricConvolution(iChannels,oChannels,3,3,3,1,1,1,1,1,1))
            conv_layers:add(nn.ReLU(true))
            iChannels = oChannels;
         end
      end
   end
   -- not update input grad
   conv_layers:get(1).gradInput = nil
   -- fully-connected layers
   local fc_layers = nn.Sequential()
   fc_layers:add(nn.View(512*1*4*4))
   fc_layers:add(nn.Linear(512*1*4*4, 4096))
   fc_layers:add(nn.ReLU(true))
   fc_layers:add(nn.Dropout(0.5))
   fc_layers:add(nn.Linear(4096, 4096))
   fc_layers:add(nn.ReLU(true))
   fc_layers:add(nn.Dropout(0.5))
   fc_layers:add(nn.Linear(4096, 487)) -- sports-1M
   -- c3d model
   local model = nn.Sequential()
   model:add(conv_layers):add(fc_layers)

   return model, {batchSize,3,16,112,112}
end

return c3d
