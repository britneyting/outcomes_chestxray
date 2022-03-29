'''
Author: Britney Ting

Main class for managing different models.
Borrowed from Ruizhi Liao with permission.
'''

import os
from tqdm import tqdm, trange
import logging
from scipy.stats import logistic
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse
import csv
from scipy.special import softmax
from scipy.special import expit
import time
import cv2

import torch
import torchvision
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model_utils import build_resnet256_6_2_1
from dataloader import build_training_dataset, build_evaluation_dataset
import eval_metrics

def build_model(model_name, checkpoint_path=None, output_channels=4):
	if checkpoint_path == None:
		if model_name == 'resnet256_6_2_1':
			model = build_resnet256_6_2_1(output_channels=output_channels)
	else:
		if model_name == 'resnet256_6_2_1':
			model = build_resnet256_6_2_1(pretrained=True,
										  pretrained_model_path=checkpoint_path,
										  output_channels=output_channels)
	return model

class ModelManager:

	def __init__(self, model_name, img_size, output_channels=4):
		self.model_name = model_name
		self.output_channels = output_channels
		self.model = build_model(self.model_name, output_channels=self.output_channels)
		self.img_size = img_size
		self.logger = logging.getLogger(__name__)

	def train(self, device, args):
		# data_dir, dataset_metadata, save_dir,
		# 	  batch_size=64, num_train_epochs=300, 
		# 	  device='cuda', init_lr=5e-4, logging_steps=50,
		# 	  label_key='edema_severity', loss_method='CrossEntropyLoss'):
		'''
		Create a logger for logging model training
		'''
		logger = logging.getLogger(__name__)

		'''
		Create an instance of traning data loader
		'''
		print('***** Instantiate a data loader *****')
        
		dataset = build_training_dataset(data_dir=args.data_dir,
										 img_size=self.img_size,
										 dataset_metadata=args.dataset_metadata,
										 label_key=args.label_key)
		data_loader = DataLoader(dataset, batch_size=args.batch_size,
								 shuffle=True, num_workers=8,
								 pin_memory=True)
		print(f'Total number of training images: {len(dataset)}')

		''' 
		Create an instance of loss
		'''
		print('***** Instantiate the training loss *****')
		if args.loss_method == 'CrossEntropyLoss':
			loss_criterion = CrossEntropyLoss().to(device)
		elif args.loss_method == 'BCEWithLogitsLoss':
			loss_criterion = BCEWithLogitsLoss().to(device)

		'''
		Create an instance of optimizer and learning rate scheduler
		'''
		print('***** Instantiate an optimizer *****')
		optimizer = optim.Adam(self.model.parameters(), lr=args.init_lr)

		'''
		Train the model
		'''
		print('***** Train the model *****')
		self.model = self.model.to(device)
		self.model.train()
		train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
		training_losses = []
		for epoch in train_iterator:
			start_time = time.time()
			epoch_loss = 0
			epoch_iterator = tqdm(data_loader, desc="Iteration")
			for i, batch in enumerate(epoch_iterator, 0):
				# Parse the batch 
                
				images, labels, image_ids = batch # NOTE: This is returned from from CXRDataset[__getitem__]
				images = images.to(device, non_blocking=True)
				labels = labels.to(device, non_blocking=True)

				# Zero out the parameter gradients
				optimizer.zero_grad()

				# Forward + backward + optimize
				outputs = self.model(images)
				pred_logits = outputs[-1]
				# Note that the logits are used here
				if args.loss_method == 'BCEWithLogitsLoss':
					labels = torch.reshape(labels, pred_logits.size())
				
				# pred_logits[labels<0] = 0
				# labels[labels<0] = 0.5
				loss = loss_criterion(pred_logits, labels.float())
				loss.backward()
				optimizer.step()

				# Record training statistics
				epoch_loss += loss.item()

				if not loss.item()>0:
					logger.info(f"loss: {loss.item()}")
					logger.info(f"pred_logits: {pred_logits}")
					logger.info(f"labels: {labels}")
			self.model.save_pretrained(args.save_dir, epoch=epoch + 1)
			interval = time.time() - start_time

			training_losses.append(epoch_loss)
			print(f'Epoch {epoch+1} finished! Epoch loss: {epoch_loss:.5f}')

			logger.info(f"  Epoch {epoch+1} loss = {epoch_loss:.5f}")
			logger.info(f"  Epoch {epoch+1} took {interval:.3f} s")

		plt.plot(range(len(training_losses)), training_losses, color='red', label='Training loss')
		plt.xlabel('Epoch')
		plt.ylabel('Training Loss')
		plt.savefig("training_loss_per_epoch.png", bbox_inches="tight")
		return

	def eval(self, device, args, checkpoint_path):
		'''
		Load the checkpoint (essentially create a "different" model)
		'''
		self.model = build_model(model_name=self.model_name,
								 output_channels=self.output_channels,
								 checkpoint_path=checkpoint_path)

		'''
		Create an instance of evaluation data loader
		'''
		print('***** Instantiate a data loader *****')
		dataset = build_evaluation_dataset(data_dir=args.data_dir,
										   img_size=self.img_size,
										   dataset_metadata=args.dataset_metadata,
										   label_key=args.label_key)
		data_loader = DataLoader(dataset, batch_size=args.batch_size,
								 shuffle=True, num_workers=8,
								 pin_memory=True)
		print(f'Total number of evaluation images: {len(dataset)}')

		'''
		Evaluate the model
		'''
		print('***** Evaluate the model *****')
		self.model = self.model.to(device)
		self.model.eval()

		# For storing labels and model predictions
		all_preds_prob = []
		all_preds_logit = []
		all_labels = []

		epoch_iterator = tqdm(data_loader, desc="Iteration")
		for i, batch in enumerate(epoch_iterator, 0):
			# Parse the batch 
			images, labels, image_ids = batch
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			
			with torch.no_grad():
				outputs = self.model(images)
				
				preds_prob = outputs[0]
				preds_logit = outputs[-1]

				if not args.label_key == 'edema_severity':
					labels = torch.reshape(labels, preds_logit.size())

				preds_prob = preds_prob.detach().cpu().numpy()
				preds_logit = preds_logit.detach().cpu().numpy()
				labels = labels.detach().cpu().numpy()

				all_preds_prob += \
					[preds_prob[j] for j in range(len(labels))]
				all_preds_logit += \
					[preds_logit[j] for j in range(len(labels))]
				all_labels += \
					[labels[j] for j in range(len(labels))]

		all_preds_class = np.argmax(all_preds_prob, axis=1)
		inference_results = {'all_preds_prob': all_preds_prob,
							 'all_preds_class': all_preds_class,
							 'all_preds_logit': all_preds_logit,
							 'all_labels': all_labels}
		eval_results = {}

		if args.label_key == 'edema_severity':
			all_onehot_labels = [convert_to_onehot(label) for label in all_labels]

			ordinal_aucs = eval_metrics.compute_ordinal_auc(all_onehot_labels, all_preds_prob)
			eval_results['ordinal_aucs'] = ordinal_aucs

			ordinal_acc_f1 = eval_metrics.compute_ordinal_acc_f1_metrics(all_onehot_labels, 
																	     all_preds_prob)
			eval_results.update(ordinal_acc_f1)

			eval_results['mse'] = eval_metrics.compute_mse(all_labels, all_preds_prob)

			results_acc_f1, _, _ = eval_metrics.compute_acc_f1_metrics(all_labels, all_preds_prob)
			eval_results.update(results_acc_f1)
		else:
			all_preds_prob = [1 / (1 + np.exp(-logit)) for logit in all_preds_logit]
			all_preds_class = np.argmax(all_preds_prob, axis=1)
			aucs = eval_metrics.compute_multiclass_auc(all_labels, all_preds_prob)
			eval_results['aucs'] = aucs

		return inference_results, eval_results

	def infer(self, device, args, checkpoint_path, img_path):
		'''
		Load the checkpoint (essentially create a "different" model)
		'''
		self.model = build_model(model_name=self.model_name,
								 output_channels=self.output_channels,
								 checkpoint_path=checkpoint_path)

		'''
		Load and Preprocess the input image
		'''
		img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)

		transform = torchvision.transforms.Compose([
			torchvision.transforms.Lambda(lambda img: img.astype(np.int16)),
			torchvision.transforms.ToPILImage(),
			torchvision.transforms.CenterCrop(args.img_size),
			torchvision.transforms.Lambda(
				lambda img: np.array(img).astype(np.float32)),
			torchvision.transforms.Lambda(
				lambda img: img / img.max())
		])

		img = transform(img)
		img = np.expand_dims(img, axis=0)
		img = np.expand_dims(img, axis=0)

		'''
		Run model inference
		'''
		self.model = self.model.to(device)
		self.model.eval()

		img = torch.tensor(img)
		img = img.to(device, non_blocking=True)
			
		with torch.no_grad():
			outputs = self.model(img)
			
			pred_logit = outputs[-1]
			pred_logit = pred_logit.detach().cpu().numpy()
			if args.label_key == 'edema_severity':
				pred_prob = outputs[0]
				pred_prob = pred_prob.detach().cpu().numpy()
			else:
				pred_prob = expit(pred_logit)

		inference_results = {'pred_prob': pred_prob,
							 'pred_logit': pred_logit}

		return inference_results