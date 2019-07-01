require 'image'
require 'nn'
require 'optim'
require 'paths'
require 'cunn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
    batchSize = 50,         -- size of a batch of images
    beta1 = 0.9,            -- beta param for adam
    dataset = 'folder',     -- folder (see data/donkey_folder.lua)
    display = 2929,
    gpu = 1,
    imgSize = 64,
    lambda = 0.002,
    loadSize = 64,
    lr = 0.02,
    name = 'celebA',
    net = 'celebA-normal',
    nIter = 2000,
    noise = 'normal',
    nz = 100,
    showEvery = 20,
    winId = 1000,
}

opt = xlua.envparams(opt)
print(opt)
--[[
if opt.display > 0 then
    display = require 'display'
    display.configure({ hostname = '0.0.0.0', port = opt.display })
else
    opt.display = false
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(opt.gpu)
end
]]
optimConfig = {
    learningRate = opt.lr,
    beta1 = opt.beta1,
}

-- load the networks
local netD = '/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/checkpoints/experiment1_25_net_D.t7'
local netG = '/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/checkpoints/experiment1_25_net_G.t7'
netG = torch.load(netG)
netD = torch.load(netD)
netD=netD:cuda()
netG=netG:cuda()



print(netG)
print(netD)
-- local L1Criterion = nn.AbsCriterion()
local L1Criterion = nn.SmoothL1Criterion()
local L2Criterion = nn.MSECriterion()
local BCECriterion = nn.BCECriterion()

-- get a ramdom batch of images to complete
--local DataLoader = paths.dofile('data/data.lua')
--local data = DataLoader.new(0, opt.dataset, opt)

local getBatch= function()
Hight_Input_size=64
Width_Input_size=64
channel_Input=3
batchSize=50
Temp_input=torch.zeros(batchSize,channel_Input,Hight_Input_size,Width_Input_size)
Temp_real_uncropped=torch.zeros(batchSize,channel_Input,Hight_Input_size,Width_Input_size)
--Train_dataset=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/Dataset_location/3-Random_rotation_angle_full_CelebA/combined_trainLR_Part.bin')
Train_dataset=torch.load('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/DataSet/2-Transformed_image/3-Testing/1-One_rotation_angle/1-R/combined_testLR.bin')

--print(Train_dataset)
--Input_Total=Train_dataset.data
--real_uncropped_Total=Train_dataset.label
counter_indices=torch.load('counter_indices.bin')
indices_Total=torch.load('indices.bin')
ind=indices_Total[counter_indices]
Temp_input:copy(Train_dataset.data:index(1,ind))
Temp_real_uncropped:copy(Train_dataset.label:index(1,ind))
counter_indices=counter_indices+1
torch.save('counter_indices.bin',counter_indices)
return Temp_real_uncropped,Temp_input

end

-- This is my Part
    nbr_samples=2000
    counter_indices=1
    new_indices={}
	
     local indices = torch.randperm(nbr_samples):long():split(opt.batchSize)
  --for i=1,torch.floor(nbr_samples/opt.batchSize) do new_indices[i]=indices[i] end
  --indices[#indices] = nil -- remove last partial batch
  torch.save('indices.bin',indices)
  torch.save('counter_indices.bin',counter_indices)
  no_batches=nbr_samples/opt.batchSize;
  count=1
  
  masked_images=torch.zeros(nbr_samples,3,64,64)
  Label_no_Transformation=torch.zeros(nbr_samples,3,64,64)
  output=torch.zeros(nbr_samples,3,64,64)
  
 for iii=1,no_batches do
      local ground_truth,images = getBatch()
     local height = images:size(3)
     local width = images:size(4)
     mask = torch.Tensor(images:size()):fill(1)
     mask:narrow(3, height / 4, height / 2):narrow(4, width / 4, width / 2):zero()
     -- sample some noise
      z = torch.Tensor(images:size(1), opt.nz, 1, 1)
      if opt.noise == 'uniform' then
      z:uniform(-1, 1)
      else
      z:normal(0, 1)
     end

     local label = torch.Tensor(images:size(1)):fill(1)

     -- convert to Cuda Tensors
     L1Criterion:cuda()
     L2Criterion:cuda()
     BCECriterion:cuda()
     mask = mask:cuda()
     images = images:cuda()
     z = z:cuda()
     label = label:cuda()

     -- Getting the  real masked images
     local masked_img = torch.cmul(images, mask)
	 
	 Label_no_Transformation:narrow(1,count,opt.batchSize):copy(ground_truth)
     masked_images:narrow(1,count,opt.batchSize):copy(masked_img)
     torch.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/Label_no_Transformation.bin',Label_no_Transformation)
     torch.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/masked_images.bin',masked_images)
     
	 
	 
	 
	 -- function performing completion
     local complete = function(masked_img, mask, z)
     local gen = netG:forward(z)
     return masked_img + torch.cmul(mask:clone():fill(1) - mask, gen)
     end

     -- closure computing df/dz and f(z)
     local loss_dL_dz = function(z)
     mlpG = netG:clone('weight', 'bias');
     mlpD = netD:clone('weight', 'bias');

     -- contextual loss
     local gen = mlpG:forward(z)
     local contextual_err = L1Criterion:forward(torch.cmul(gen, mask), masked_img)
     local df_do_con = L1Criterion:backward(torch.cmul(gen, mask), masked_img)
     -- perceptual loss
     local pred = mlpD:forward(gen)
     local perceptual_err = BCECriterion:forward(pred, label)
     local df_do_per = BCECriterion:backward(pred, label)
     local dD_dz = mlpD:updateGradInput(gen, df_do_per)
     local grads = mlpG:updateGradInput(z, torch.cmul(df_do_con, mask) + opt.lambda * dD_dz)
    -- sum
     local err = contextual_err + opt.lambda * perceptual_err
     -- print(err, contextual_err, perceptual_err)
     return err, grads
     end
	 
	 
     local save_dir = 'completed/' .. opt.name
     paths.mkdir(save_dir)
      if display then
     display.image(images, { win = opt.winId, title = "original images" })
     display.image(masked_img, { win = opt.winId + 1, title = "masked images" })
     display.image(complete(masked_img, mask, z), { win = opt.winId + 2, title = "inpainted images" })
     end

     --image.save(save_dir .. '/image_000_batch_no.jpg', image.toDisplayTensor({ input = masked_img, nrow = 10 }))
     image.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/10_masked_images/Masked_image_batch_no'..iii..'.png', image.toDisplayTensor({ input = masked_img, nrow = 10 }))
	 image.save(save_dir .. '/ground_truth_image_000_batch_no.jpg', image.toDisplayTensor({ input = ground_truth, nrow = 10 }))
     image.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/10_Ground_Truth/ground_truth_image_batch_no'..iii..'.png', image.toDisplayTensor({ input = ground_truth, nrow = 10 }))
	 
      	  
		 print("Testing is Runing on Batch= " .. iii .. ".")
		 for iter = 1, opt.nIter do
         z = optim.adam(loss_dL_dz, z, optimConfig)
	     if opt.noise == 'uniform' then 
	     z = z:clamp(-1, 1)                                
	     end
		 
          xlua.progress(iter , opt.nIter )
    
end

-- display in browser and save images
               local gen = netG:forward(z)
               local masked_gen = torch.cmul(gen, mask)
               local comp_img = complete(masked_img, mask, z)
	          if display then
              display.image(comp_img, { win = opt.winId + 2, title = "inpainted images" })
              display.image(masked_gen, { win = opt.winId + 4, title = "masked generated images" })
              display.image(gen, { win = opt.winId + 3, title = "generated images" })
              end
--image.save((save_dir .. '/image_%03d_batch.jpg'):format(iter / opt.showEvery), image.toDisplayTensor({ input = comp_img, nrow = 10 }))
image.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/10_completed_images/Completed_images_batch'..iii..'.png', image.toDisplayTensor({ input = comp_img, nrow = 10 }))
image.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/10_gen_images/Generated_images_batch'..iii..'.png', image.toDisplayTensor({ input = gen, nrow = 10 }))
image.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/10_masked_gen_images/Masked_Generated_images_batch'..iii..'.png', image.toDisplayTensor({ input = masked_gen, nrow = 10 }))	
output:narrow(1,count,opt.batchSize):copy(comp_img)
torch.save('/data/hossamkasem/4-GAN_DFTN/2-Image_inpainting/1-R/1-one_rotation_angle/1-Celeb_A/1-Original_DC_image_inpainting/Testing_results/output.bin',output)
count=count+opt.batchSize 

end


--print("Completed images have been saved into " .. save_dir .. ".")
