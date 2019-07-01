--[[
    This file is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/main.lua).

]]--

require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'model'
require 'cunn'
require'cudnn'

opt = {
   dataset = 'CelebA',     -- folder
   batchSize = 50,         -- #  of images per batch
   loadSize = 64,          -- rescale images to this value
   imgSize = 64,           -- size of the images
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 0,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 5562,         -- display port during training. 0 = no display
   winId = 10,             -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',   -- name of your experiment, to save the networks
   Image_name='/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/pics/Inpainting',
   noise = 'normal',       -- uniform / normal
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
--local DataLoader = paths.dofile('data/data.lua')
--local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
data_Size=42950
print("Dataset: " .. opt.dataset, " Size: ", data_Size)
----------------------------------------------------------------------------
local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

-- load models
--[[
local netG = get_netG(nz, ngf, nc)
local netD = get_netD(nc, ndf)
]]

netG=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/checkpoints/saved_model_dataset_42950/experiment1_4_net_G.t7')
netD=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/checkpoints/saved_model_dataset_42950/experiment1_4_net_D.t7')

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.imgSize, opt.imgSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
    input = input:cuda()
    noise = noise:cuda()
    label = label:cuda()
    netD:cuda()
    netG:cuda()
    criterion:cuda()
    cudnn.benchmark = true
    cudnn.convert(netG, cudnn)
    cudnn.convert(netD, cudnn)
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
--[[
if opt.display then
    disp = require 'display'
    disp.configure({ hostname='0.0.0.0', port=opt.display })
end
]]

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end



local getBatch= function()
Hight_Input_size=64
Width_Input_size=64
channel_Input=3
batchSize=50
Temp_input=torch.zeros(batchSize,channel_Input,Hight_Input_size,Width_Input_size)
Temp_real_uncropped=torch.zeros(batchSize,channel_Input,Hight_Input_size,Width_Input_size)
--Train_dataset=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/DataSet/1-Original_Dataset/Train/original_CelebA_64_64.bin')
Train_dataset=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/DataSet/1-CelebA/2-Transformed_image/1-One angle/1-R/Train/42952/train_dataset.bin')

--print(Train_dataset)
--Input_Total=Train_dataset.data
--real_uncropped_Total=Train_dataset.label
counter_indices=torch.load('counter_indices.bin')
indices_Total=torch.load('indices.bin')
ind=indices_Total[counter_indices]
Temp_input:copy(Train_dataset.data:index(1,ind))
Temp_real_uncropped:copy(Train_dataset.data:index(1,ind))
counter_indices=counter_indices+1
torch.save('counter_indices.bin',counter_indices)
return Temp_real_uncropped

end
-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   
   --%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   -- This is my Part
   nbr_samples=42950
   --nbr_samples=100
   counter_indices=1
   new_indices={}
  local indices = torch.randperm(nbr_samples):long():split(opt.batchSize)
  --for i=1,torch.floor(nbr_samples/opt.batchSize) do new_indices[i]=indices[i] end
  --indices[#indices] = nil -- remove last partial batch
  torch.save('indices.bin',indices)
  torch.save('counter_indices.bin',counter_indices)
  --%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
   for i = 1, math.min(42950, opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 2 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          --local real = getBatch()
		  --image.save(opt.Image_name..counter..'_real.png',real[1])
		  image.save(opt.Image_name..counter..'_fake.png',fake[1])
          --disp.image(fake, {win=opt.winId, title=opt.name})
          --disp.image(real, {win=opt.winId * 3, title=opt.name})
      end

      -- logging
      if ((i - 1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i - 1) / opt.batchSize),
                 math.floor(math.min(42950, opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
