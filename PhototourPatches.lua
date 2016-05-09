-- Loads the Phototour Patches dataset and saves it in Torch7 format for faster loading.
-- Vassileios Balntas v.balntas15@imperial.ac.uk

-- How to use
-- 1) Download original dataset from http://phototour.cs.washington.edu/patches/default.htm
-- 2) Edit the opts below to match your specific details
-- 3) Run the code with th PhototourPatches.lua -> enjoy .t7 output

-- You can also download the converted .t7 files directly using
-- wget http://www.iis.ee.ic.ac.uk/~vb2415/notredame-t7.tar.gz
-- wget http://www.iis.ee.ic.ac.uk/~vb2415/liberty-t7.tar.gz
-- wget http://www.iis.ee.ic.ac.uk/~vb2415/yosemite-t7.tar.gz

require 'torch'
require 'xlua'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

-- options - change to match your folders and your images - e.g I 
-- converted them to pgm some time ago so probably yours will be .jpeg 
opt  = {}
opt.name = 'liberty'
opt.dir = '/home/vassilis/Datasets/'..opt.name
opt.ext = 'pgm'
opt.ids = opt.dir..'/info.txt'
opt.gt100k = opt.dir..'/m50_100000_100000_0.txt'
opt.gt500k = opt.dir..'/m50_500000_500000_0.txt'

-- read the ids gt
ids = {}
local file = io.open(opt.ids)
if file then
    for line in file:lines() do
       local idx,dummy  = unpack(line:split(" "))
       table.insert(ids,idx)
    end
else
end
ids = torch.Tensor(ids)
npatches =  (ids:size(1))
print(npatches)

-- read the 100k gt - test data
gt100k = {}
local file = io.open(opt.gt100k)
if file then
    for line in file:lines() do
       l = line:split(" ")
       table.insert(gt100k,l)
    end
else
end
gt100k = torch.Tensor(gt100k)

-- read the 500k gt - train data
gt500k = {}
local file = io.open(opt.gt500k)
if file then
    for line in file:lines() do
       l = line:split(" ")
       table.insert(gt500k,l)
    end
else
end
gt500k = torch.Tensor(gt500k)

-- load the files 
files = {}
for file in paths.files(opt.dir) do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      table.insert(files, paths.concat(opt.dir,file))
   end
end

-- check for errors
if #files == 0 then
   error('check that the directory contains .' .. opt.ext..' files.')
end

-- alphabetical sort of the img files
table.sort(files, function (a,b) return a < b end)

idx = 1
-- save both 64 and 32 versions to avoid rescaling on-the-fly during experiments
patches64 = torch.Tensor(npatches,1,64,64)
patches32 = torch.Tensor(npatches,1,32,32)
-- Go over the file list:
for i,file in ipairs(files) do
   print(file)
   -- load each image and split to patches
   im = image.load(file)
   -- print(im:nDimension())
   -- print(im:size())
   for i=1,1024,64 do
      for j=1,1024,64 do
	 p = im[{ {},{i,i+63}, {j,j+63} }]
	 patches64[{ {idx},{},{},{} }] = p
	 psc = image.scale(p,32,32)
	 patches32[{ {idx},{},{},{} }] = psc
	 idx = idx + 1
	 if idx==npatches then break end
      end
      if idx==npatches then break end
   end
end

-- done & save 
dataset = {}
dataset.patches64 = patches64
dataset.patches32 = patches32
dataset.labels = ids:int()
dataset.gt100k = gt100k:int()
dataset.gt500k = gt500k:int()
torch.save(opt.name..".t7", dataset)
