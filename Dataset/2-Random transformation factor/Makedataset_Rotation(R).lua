-- this program is used to generate a rotated train dataset using Affine transform and bilinear sampler

require 'image';
require 'stn';



--===============================================================================================
-- new program to generate the rotate data set
--===============================================================================================

train_data = torch.load('/data/hossamkasem/3-DFTN_Exper._CVPR_TNNL/6-Missing_Parts_Exp/Dataset/SVHN/original_dataset_54_54/train_dataset.bin')
testimg=train_data.data
label=train_data.data
S4=testimg:size()[3]
cahnnel=testimg:size()[2]
im2=torch.zeros(cahnnel,S4/1,S4/1)
im=torch.zeros(cahnnel,64,64)
output_2=torch.zeros(111000,cahnnel,64,64)
label_temp=torch.zeros(cahnnel,S4,S4)
label_temp_scaled=torch.zeros(cahnnel,64,64)
Final_label=torch.zeros(111000,cahnnel,64,64)
Random_angle=torch.zeros(4,1);
Random_angle[1]=-35;
Random_angle[2]=-20;
Random_angle[3]=10;
Random_angle[4]=15;
a = 1;
b = 5;
local height = 64
local width = 64
mask = torch.Tensor(cahnnel,64,64):fill(1)
mask[{{},{16,48},{16,48}}]=0
for iii = 1,111000 do
im2 = testimg[iii]
im = image.scale(im2, 64,64, 'bicubic')
masked_img = torch.cmul(im:double(), mask:double())
im=masked_img
label_temp=label[iii]
label_temp_scaled=image.scale(label_temp,64,64, 'bicubic')
Final_label[iii]=label_temp_scaled
width = im:size()[3]  -- 512 / 4
height = im:size()[2]  -- 512 / 4
nchan = im:size()[1]  -- 3

grid_y = torch.ger( torch.linspace(-1,1,height), torch.ones(width) )
grid_x = torch.ger( torch.ones(height), torch.linspace(-1,1,width) )

flow = torch.FloatTensor()
flow:resize(2,height,width)
flow:zero()

-- Apply uniform scale
flow_scale = torch.FloatTensor()
flow_scale:resize(2,height,width)
flow_scale[1] = grid_y
flow_scale[2] = grid_x
flow_scale[1]:add(1):mul(0.5) -- 0 to 1
flow_scale[2]:add(1):mul(0.5) -- 0 to 1
flow_scale[1]:mul(height-1)
flow_scale[2]:mul(width-1)
flow:add(flow_scale)

flow_rot = torch.FloatTensor()
flow_rot:resize(2,height,width)
flow_rot[1] = grid_y * ((height-1)/2) * -1
flow_rot[2] = grid_x * ((width-1)/2) * -1
view = flow_rot:reshape(2,height*width)

random_number = (b-a)*torch.rand(1) + a
r_floor=torch.floor(random_number)
x=r_floor[1]
rot_angle = Random_angle[x]  -- a nice non-integer value
r_temp= rot_angle/180*math.pi
r=r_temp[1]
rotmat = torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
flow_rotr = torch.mm(rotmat, view)
flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
flow:add(flow_rot)
im_bicubic = image.warp(im:double(), flow:double(), 'bicubic', false)
output_2[iii]=im_bicubic
print(iii)
end
dump2={}
dump2.data= output_2:float()
dump2.label=Final_label:float()
torch.save('combined_trainLR.bin',dump2)
v1=dump2.data[1000]
v2=dump2.label[1000]
image.save('original.png',image.toDisplayTensor(v2))
image.save('Transformed.png',image.toDisplayTensor(v1))