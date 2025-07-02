import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.utils.data as Data
import h5py
import numpy as np
import torch
import torchvision.transforms as transforms


class H5Dataset_withtext(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['A_patchs'].keys())
        h5f.close()
        self.transform = transforms.Compose([
             transforms.Resize((256,  256)), 
            ])       

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        imageA = np.array(h5f['A_patchs'][key])  #ir
        imageB = np.array(h5f['B_patchs'][key])  #vis
        mask = np.array(h5f['mask_patchs'][key]) #msk
        text =  torch.from_numpy(np.array(h5f['text_patchs'][key])) #text

        h5f.close()
        
        imageA, imageB , mask = torch.Tensor(imageA), torch.Tensor(imageB), torch.Tensor(mask)
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)
        mask = self.transform(mask)

    
        return imageA, imageB, mask, text

class H5Dataset_nofusion(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['A_patchs'].keys())
        h5f.close()

        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        
        self.transform = transforms.Compose([
             transforms.Resize((256, 256)),  # CRIS 416
            ])       
        self.transform2 = transforms.Compose([
             transforms.Resize((416, 416)),  # CRIS 416
            ])     
    def __len__(self):
        return len(self.keys)
    
    def convert(self, imageA, imageB, mask=None,):
        # ImageA ToTensor & Normalize
        # imageA = torch.from_numpy(imageA.transpose((2, 0, 1)))
        if not isinstance(imageA, torch.FloatTensor):
            imageA = imageA.float()
        imageA = imageA.div_(255.) #.sub_(self.mean).div_(self.std)

        # ImageB ToTensor & Normalize
        # imageB = torch.from_numpy(imageB.transpose((2, 0, 1)))
        if not isinstance(imageB, torch.FloatTensor):
            imageB = imageB.float()
        imageB = imageB.div_(255.) #.sub_(self.mean).div_(self.std)



        # Mask ToTensor
        if mask is not None:
            # mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        imageA = np.array(h5f['A_patchs'][key])  #ir
        imageB = np.array(h5f['B_patchs'][key])  #vis
        mask = np.array(h5f['mask_patchs'][key]) #msk
        text =  torch.from_numpy(np.array(h5f['text_patchs'][key])) #text

        h5f.close()
        
        imageA, imageB , mask = torch.FloatTensor(imageA), torch.FloatTensor(imageB), torch.FloatTensor(mask)

        
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)
        mask = self.transform(mask)
        # imageA, imageB, mask = self.convert(imageA, imageB, mask)

        return imageA, imageB, mask, text
    
    
    
class H5Dataset_fusion(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['A_patchs'].keys())
        h5f.close()

        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)

        
        self.transform = transforms.Compose([
             transforms.Resize((416, 416)),  # CRIS 416
            ])       

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        imageA = np.array(h5f['A_patchs'][key])  #ir
        imageB = np.array(h5f['B_patchs'][key])  #vis
        mask = np.array(h5f['mask_patchs'][key]) #msk
        text =  torch.from_numpy(np.array(h5f['text_patchs'][key])) #text
        fusion = np.array(h5f['fusion_patchs'][key]) #fusion

        h5f.close()
        
        imageA, imageB , mask , fusion= torch.FloatTensor(imageA), torch.FloatTensor(imageB), torch.FloatTensor(mask), torch.FloatTensor(fusion)
        imageA = self.transform(imageA)
        imageB = self.transform(imageB)
        mask = self.transform(mask)
        fusion = self.transform(fusion)
        # imageA, imageB, fusion, mask = self.convert(imageA, imageB, fusion, mask)

        return imageA, imageB, mask, text, fusion
    
    def convert(self, imageA, imageB, fusion, mask=None,):
        # ImageA ToTensor & Normalize
        # imageA = torch.from_numpy(imageA.transpose((2, 0, 1)))
        if not isinstance(imageA, torch.FloatTensor):
            imageA = imageA.float()
        imageA = imageA.div_(255.) #.sub_(self.mean).div_(self.std)

        # ImageB ToTensor & Normalize
        # imageB = torch.from_numpy(imageB.transpose((2, 0, 1)))
        if not isinstance(imageB, torch.FloatTensor):
            imageB = imageB.float()
        imageB = imageB.div_(255.) #.sub_(self.mean).div_(self.std)

        # Fusion ToTensor & Normalize
        # fusion = torch.from_numpy(fusion.transpose((2, 0, 1)))
        if not isinstance(fusion, torch.FloatTensor):
            fusion = fusion.float()
        fusion = fusion.div_(255.) #.sub_(self.mean).div_(self.std)

        # Mask ToTensor
        if mask is not None:
            # mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()

        return imageA, imageB, fusion, mask
    



# class H5Dataset_fusion(Data.Dataset):
#     def __init__(self, h5file_path):
#         self.h5file_path = h5file_path
#         h5f = h5py.File(h5file_path, 'r')
#         self.keys = list(h5f['A_patchs'].keys())
#         h5f.close()

#         self.mean = torch.tensor([0.48145466, 0.4578275,
#                                   0.40821073]).reshape(3, 1, 1)
        
#         self.std = torch.tensor([0.26862954, 0.26130258,
#                                  0.27577711]).reshape(3, 1, 1)

        
#         self.transform = transforms.Compose([
#              transforms.Resize((416, 416)),  # CRIS 416
#             ])       

#     def __len__(self):
#         return len(self.keys)
    
#     def __getitem__(self, index):
#         h5f = h5py.File(self.h5file_path, 'r')
#         key = self.keys[index]
#         imageA = np.array(h5f['A_patchs'][key])  #ir
#         imageB = np.array(h5f['B_patchs'][key])  #vis
#         mask = np.array(h5f['mask_patchs'][key]) #msk
#         text =  torch.from_numpy(np.array(h5f['text_patchs'][key])) #text
#         fusion = np.array(h5f['fusion_patchs'][key]) #fusion

#         h5f.close()
        
#         imageA, imageB , mask , fusion= torch.Tensor(imageA), torch.Tensor(imageB), torch.Tensor(mask), torch.Tensor(fusion)
#         imageA = self.transform(imageA)
#         imageB = self.transform(imageB)
#         mask = self.transform(mask)
#         fusion = self.transform(fusion)
#         imageA, imageB, fusion, mask = self.convert(imageA, imageB, fusion, mask)

#         return imageA, imageB, mask, text, fusion
    
#     def convert(self, imageA, imageB, fusion, mask=None,):
#         # ImageA ToTensor & Normalize
#         # imageA = torch.from_numpy(imageA.transpose((2, 0, 1)))
#         if not isinstance(imageA, torch.FloatTensor):
#             imageA = imageA.float()
#         imageA = imageA.repeat(3,1,1).div_(255.).sub_(self.mean).div_(self.std)

#         # ImageB ToTensor & Normalize
#         # imageB = torch.from_numpy(imageB.transpose((2, 0, 1)))
#         if not isinstance(imageB, torch.FloatTensor):
#             imageB = imageB.float()
#         imageB = imageB.div_(255.).sub_(self.mean).div_(self.std)

#         # Fusion ToTensor & Normalize
#         # fusion = torch.from_numpy(fusion.transpose((2, 0, 1)))
#         if not isinstance(fusion, torch.FloatTensor):
#             fusion = fusion.float()
#         fusion = fusion.div_(255.) #.sub_(self.mean).div_(self.std)

#         # Mask ToTensor
#         if mask is not None:
#             # mask = torch.from_numpy(mask)
#             if not isinstance(mask, torch.FloatTensor):
#                 mask = mask.float()

#         return imageA, imageB, fusion, mask