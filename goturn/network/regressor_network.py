import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = False

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1)
                    # padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1)
                    # padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv1 = nn.Conv2d(3, 96, 11, 4)
		self.lrn1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv2 = nn.Conv2d(96, 256, 5, padding=0, groups=2)
		self.lrn2 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
		self.lrn3 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)
		self.lrn4 = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
		self.lrn5 = LRN(local_size=5, alpha=0.0001, beta=0.75)

		self.conv1_p = nn.Conv2d(3, 96, 11, 4)
		self.lrn1_p = LRN(local_size=5, alpha=0.0001, beta=0.75)
		self.conv2_p = nn.Conv2d(96, 256, 5, padding=0, groups=2)
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
		self.lstm = nn.LSTM(124, 124, 1)
		self.fc9 = nn.Linear(124, 4)

	def forward(self, img, ref, hidden):
		seq_len = img.shape[0]
		img_output = self.lrn1(F.max_pool2d(F.relu(self.conv1(img)), 3, 2))
		img_output = self.lrn2(F.max_pool2d(F.relu(self.conv2(img_output)), 3, 2))
		img_output = self.lrn3(F.max_pool2d(F.relu(self.conv3(img_output)), 3, 2))
		img_output = self.lrn4(F.max_pool2d(F.relu(self.conv4(img_output)), 3, 2))
		img_output = self.lrn5(F.max_pool2d(F.relu(self.conv5(img_output)), 3, 2))

		ref_output = self.lrn1(F.max_pool2d(F.relu(self.conv1(ref)), 3, 2))
		ref_output = self.lrn2(F.max_pool2d(F.relu(self.conv2(ref_output)), 3, 2))
		ref_output = self.lrn3(F.max_pool2d(F.relu(self.conv3(ref_output)), 3, 2))
		ref_output = self.lrn4(F.max_pool2d(F.relu(self.conv4(ref_output)), 3, 2))
		ref_output = self.lrn5(F.max_pool2d(F.relu(self.conv5(ref_output)), 3, 2))

		output = torch.cat(img_output.view(seq_len, 1, 18432/2), ref_output.view(seq_len, 1, 18432/2), axis=2)

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
		data_files = os.listdir(input_dir+'/data/')
		for file in data_files:
			inp = self.load_data(input_dir, file)
			bbox_labels = self.load_label(input_dir+'/labels/'+file[:-4]+'groundtruth.txt')
			ground_truth = np.zeros(inp.shape[0], 4)
			for i, label in enumerate(bbox_labels):
				ground_truth[i] = np.array([label.x1, label.y1, label.x2, label.y2])
			ground_truth = Variable(torch.from_numpy(ground_truth), requires_grad=False)
			hidden_0 = self.initHidden()
			pred = self.forward(inp[1:], inp[:-1], hidden_0)
			loss = self.loss(pred, ground_truth)
			if grad:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			else:
				total_loss+=loss
		return total_loss

	def train(self, input_dir, epochs=20):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		for iter in epochs:
			self.fit(input_dir)
			train_loss = self.fit(input_dir, False)
			print('Loss epoch %i is %f' %(iter, train_loss))
			if train_loss>prev_loss:
				with open('tracker.pt', 'w') as f:
					torch.save(model.state_dict(), f)

	def load_data(self, input_dir, file):
		images = pickle.load(input_dir+'/data/'+file)
		cropped_images = np.zeros(images.shape[0], )
		bbox_labels = self.load_label(input_dir+'/labels/'+file[:-4]+'groundtruth.txt')
		for n in range(images.shape[0]):
			search_region, _, _,  _ = cropPadImage(bbox_labels[i], images[i])
			cropped_images[i] = search_region
		return cropped_images

	def load_label(self, file):
		with open(file, 'r') as f:
			data = f.read()
		labels = [map(float, line.split(',')) for line in data.split('\n')[:-1]]
		bbox_labels = []
		for i in range(len(labels)):
			x1 = min(labels[i][0], min(labels[i][2], min(labels[i][4], labels[i][6]))) - 1
			y1 = min(labels[i][1], min(labels[i][3], min(labels[i][5], labels[i][7]))) - 1
			x2 = max(labels[i][0], max(labels[i][2], max(labels[i][4], labels[i][6]))) - 1
			y2 = max(labels[i][1], max(labels[i][3], max(labels[i][5], labels[i][7]))) - 1
			bbox = BoundingBox(x1, y1, x2, y2)
			bbox_labels.append(bbox)
		return bbox_labels

def main():
	model = Model()
	model.load_state_dict(torch.load("/home/manpk/Desktop/vision/project/PY-GOTURN/tracker.init.pt"))
	input_dir = "/home/manpk/Desktop/vision/project/vot2014/"
	model.train(input_dir)


if __name__ == "__main__":
	main()