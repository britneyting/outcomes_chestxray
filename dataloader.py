'''
Author: Britney Ting

Main class for managing different datasets.
Borrowed from Ruizhi Liao with permission.
'''

from dataloader_utils import CXRImageDataset
import numpy as np
import torchvision

def build_training_dataset(data_dir, img_size: int, dataset_metadata,
						   random_degrees=[-20,20], random_translate=[0.1,0.1], label_key='readmission_time'):
	transform=torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
		torchvision.transforms.ToPILImage(),
		torchvision.transforms.RandomAffine(degrees=random_degrees, translate=random_translate),
		torchvision.transforms.CenterCrop(img_size),
		torchvision.transforms.Lambda(
			lambda img: np.array(img).astype(np.float32)),
		torchvision.transforms.Lambda(
			lambda img: img / img.max())
	])
	training_dataset = CXRImageDataset(data_dir=data_dir, 
									   dataset_metadata=dataset_metadata, 
									   transform=transform,
									   label_key=label_key)

	return training_dataset

def build_evaluation_dataset(data_dir, img_size: int, dataset_metadata, label_key='readmission_time'):
	transform=torchvision.transforms.Compose([
		torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
		torchvision.transforms.ToPILImage(),
		torchvision.transforms.CenterCrop(img_size),
		torchvision.transforms.Lambda(
			lambda img: np.array(img).astype(np.float32)),
		torchvision.transforms.Lambda(
			lambda img: img / img.max())
	])
	evaluation_dataset = CXRImageDataset(data_dir=data_dir, 
									     dataset_metadata=dataset_metadata, 
									     transform=transform,
									     label_key=label_key)

	return evaluation_dataset