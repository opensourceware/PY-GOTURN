import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle

use_cuda = False

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1)
                    # padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            d = x.pow(2).unsqueeze(1)
            d = self.average(d).squeeze(1)
            d = d.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            d = x.pow(2)
            d = self.average(d)
            d = d.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(d)
        return x


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(3, 96, 11, 4)
		self.lrn1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
		self.lrn2 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
		self.lrn3 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
		self.lrn4 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
		self.lrn5 = LRN(local_size=5, alpha=0.0001, beta=0.75)

		self.conv1_p = nn.Conv2d(3, 96, 11, 4)
		self.lrn1_p = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv2_p = nn.Conv2d(96, 256, 5, padding=2, groups=2)
		self.lrn2_p = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv3_p = nn.Conv2d(256, 384, 3, padding=1)
		self.lrn3_p = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv4_p = nn.Conv2d(384, 384, 3, padding=1, groups=2)
		self.lrn4_p = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv5_p = nn.Conv2d(384, 256, 3, padding=1, groups=2)
		self.lrn5_p = LRN(local_size=5, alpha=0.0001, beta=0.75)

		self.fc6 = nn.Linear(18432, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc7b = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, 124)
		self.hidden_size = 124
		self.lstm = nn.LSTM(124, self.hidden_size, 1)
		self.fc9 = nn.Linear(124, 4)

	def forward(self, img, ref, hidden):
		# seq_len = img.shape[0]
		img_output = self.lrn1(F.max_pool2d(F.relu(self.conv1(img)), 3, 2))
		# print(img_output.size())
		img_output = self.lrn2(F.max_pool2d(F.relu(self.conv2(img_output)), 3, 2))
		# print(img_output.size())
		img_output = self.lrn3(F.relu(self.conv3(img_output)))
		# print(img_output.size())
		img_output = self.lrn4(F.relu(self.conv4(img_output)))
		# print(img_output.size())
		img_output = self.lrn5(F.relu(self.conv5(img_output)))
		# print(img_output.size())

		ref_output = self.lrn1(F.max_pool2d(F.relu(self.conv1(ref)), 3, 2))
		ref_output = self.lrn2(F.max_pool2d(F.relu(self.conv2(ref_output)), 3, 2))
		ref_output = self.lrn3(F.relu(self.conv3(ref_output)))
		ref_output = self.lrn4(F.relu(self.conv4(ref_output)))
		ref_output = self.lrn5(F.relu(self.conv5(ref_output)))

		output = torch.cat(img_output.view(self.seq_len, 1, 18432/2), ref_output.view(self.seq_len, 1, 18432/2), axis=2)

		output = nn.dropout(F.relu(self.fc6(output)))
		output = nn.dropout(F.relu(self.fc7(output)))
		output = nn.dropout(F.relu(self.fc7b(output)))
		output = nn.dropout(F.relu(self.fc8(output)))
		output, hidden = self.lstm(output, hidden)
		output = self.fc9(output)
		return output

	def initHidden(self):
		result = Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

	def loss(self, pred, ground_truth):
		return torch.nn.MSELoss(size_average=False)(pred, ground_truth)

	def fit(self, input_dir, grad=True):
		total_loss = 0
		with open(input_dir+'list.txt', 'r') as f:
			data_files = f.read().split('\n')[:-1]

		for file in data_files:
			inp, target, ground_truth = self.load_data(input_dir, file)
			self.seq_len = inp.shape[0]
			inp = Variable(torch.from_numpy(inp[1:]).type(torch.FloatTensor), requires_grad=False)
			target = Variable(torch.from_numpy(target[:-1]).type(torch.FloatTensor), requires_grad=False)
			ground_truth = Variable(torch.from_numpy(ground_truth).type(torch.FloatTensor), requires_grad=False)
			hidden_0 = self.initHidden()
			pred = self.forward(inp, target, hidden_0)
			loss = self.loss(pred, ground_truth)
			if grad:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			else:
				total_loss+=loss
			print(file)
			break
		return total_loss

	def train(self, input_dir, epochs=20):
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		prev_loss = np.inf
		for iter in range(epochs):
			self.fit(input_dir)
			train_loss = self.fit(input_dir, False)
			print('Loss epoch %i is %f' %(iter, train_loss))
			if train_loss<prev_loss:
				with open('tracker.pt', 'w') as f:
					torch.save(model.state_dict(), f)
			prev_loss = train_loss

	def load_data(self, input_dir, file):
		with open(input_dir+file+".search.pkl", "r") as f:
			cropped_images = pickle.load(f)
		with open(input_dir+file+".target.pkl", "r") as f:
			cropped_target = pickle.load(f)
		with open(input_dir+file+".groundtruth.pkl", "r") as f:
			ground_truth = pickle.load(f)
		return cropped_images, cropped_target, ground_truth

def main():
	model = Model()
	model.load_state_dict(torch.load("/home/manpk/Desktop/vision/project/PY-GOTURN/tracker.init.pt"))
	input_dir = "/home/manpk/Desktop/vision/project/d/"
	model.train(input_dir)


if __name__ == "__main__":
	main()