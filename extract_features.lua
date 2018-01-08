require 'cudnn'
require 'nn'
require 'cunn'
npy4th = require 'npy4th'
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-inputdir', '', 'Path to video frame dir')
cmd:option('-outdir', '', 'Path to output feature dir')
cmd:option('-modelfile', '', 'Path to model file')
cmd:option('-batchsize', 64, 'input batch size')
cmd:text()
local opt = cmd:parse(arg or {})
-- local c3d = require 'c3d'
-- local model = c3d()
local model = torch.load(opt.modelfile)
model:evaluate() -- turn on evaluation model (e.g., disable dropout)
model:cuda() -- ship model to GPU
local input = nil
local mean = npy4th.loadnpy('sports1m-mean.npy'):float() -- (1, 3, 16, 128, 171), double2float
local start = 0
while true do
    if not paths.filep(opt.inputdir .. string.format('%05d.jpg', start + 16)) then
        break
    end
    local sample = torch.FloatTensor(1, 16, 3, 128, 171)
    for i = 1, 16 do
        local image_path = opt.inputdir .. string.format('/%05d.jpg', start + i)
        local img = image.load(image_path, 3, 'byte') -- c * h * w, rgb, 3 and 'byte' is necessary
        img = image.scale(img, 171, 128)
        sample[{1, i, {}, {}, {}}] = img:float()
    end
    input = input ~= nil and torch.cat(input, sample, 1) or sample
    start = start + 8
end
input = input:transpose(2, 3)
input = input - mean:expandAs(input) -- subtract mean
input = input:narrow(4, 8, 112):narrow(5, 29, 112)
print(input:size())
local fc6_features = nil
local b = 1
while true do
    local s = (b - 1) * opt.batchsize + 1
    local e = math.min(input:size(1), b * opt.batchsize) 
    input_split = input:narrow(1, s, e - s + 1) 
    model:forward(input_split:cuda())
    fc6_out = model:get(2):get(2).output:float()
    fc6_features = fc6_features ~= nil and torch.cat(fc6_features, fc6_out, 1) or fc6_out
    if e == input:size(1) then
        break
    end
    b = b + 1
end
out_fn = paths.concat(opt.outdir, paths.basename(opt.inputdir) .. '.npy')
npy4th.savenpy(out_fn, fc6_features)
