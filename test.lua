require 'cudnn'
require 'nn'
require 'cunn'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-data', '', 'Path to video frame dir')
cmd:text()
local opt = cmd:parse(arg or {})
require 'image'
-- local c3d = require 'c3d'
-- local model = c3d()
local model = torch.load('c3d_sports1m_it1900000.t7')
model:evaluate() -- turn on evaluation model (e.g., disable dropout)
model:cuda() -- ship model to GPU
local input = torch.CudaTensor(10, 3, 16, 112, 112)
for b = 1, 10 do
    for i = 1, 16 do
        local image_path = opt.data .. string.format('/%06d.jpg', 16 * (b - 1) + i)
        local img = image.load(image_path, 3, 'byte') -- n * h * w, rgb, 3 and 'byte' is necessary
        img = image.scale(img, 171, 128)
        img = image.crop(img, 0, 0, 112, 112)
        img = img:float()--:cuda()
        input[b][1][i] = img[1]
        input[b][2][i] = img[2]
        input[b][3][i] = img[3]
    end
end

prob = model:forward(input):float()
