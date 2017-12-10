import argparse
import setproctitle
from goturn.logger.logger import setup_logger
from goturn.network.regressor import regressor
from goturn.loader.loader_vot import loader_vot
from goturn.tracker.tracker import tracker
from goturn.tracker.tracker_manager import tracker_manager
from goturn.loader.video import video
import torch
from torch import nn
import torch.nn.functional as F


setproctitle.setproctitle('SHOW_TRACKER_VOT')
logger = setup_logger(logfile=None)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "Path to the prototxt")
ap.add_argument("-m", "--model", required = True, help = "Path to the model")
ap.add_argument("-v", "--input", required = True, help = "Path to the vot directory")
ap.add_argument("-g", "--gpuID", required = False, help = "gpu to use")
args = vars(ap.parse_args())

do_train = False
objRegressor = regressor(args['prototxt'], args['model'], 1, do_train, logger)
# print("Stop here")
# objTracker = tracker(False, logger) # Currently no idea why this class is needed, eventually we shall figure it out
# objLoaderVot = loader_vot(args['input'], logger)

# videos = objLoaderVot.get_videos()

# objTrackerVis = tracker_manager(videos, objRegressor, objTracker, logger)
# objTrackerVis.trackAll(0, 1)

#Torch model
model = regressor_network.Model()

conv1_w = objRegressor.net.params['conv1_p'][0].data
conv2_w = objRegressor.net.params['conv2_p'][0].data
conv3_w = objRegressor.net.params['conv3_p'][0].data
conv4_w = objRegressor.net.params['conv4_p'][0].data
conv5_w = objRegressor.net.params['conv5_p'][0].data
fc6_w = objRegressor.net.params['fc6-new'][0].data
fc7_w = objRegressor.net.params['fc7-new'][0].data
fc7b_w = objRegressor.net.params['fc7-newb'][0].data

conv1_b = objRegressor.net.params['conv1_p'][1].data
conv2_b = objRegressor.net.params['conv2_p'][1].data
conv3_b = objRegressor.net.params['conv3_p'][1].data
conv4_b = objRegressor.net.params['conv4_p'][1].data
conv5_b = objRegressor.net.params['conv5_p'][1].data
fc6_b = objRegressor.net.params['fc6-new'][1].data
fc7_b = objRegressor.net.params['fc7-new'][1].data
fc7b_b = objRegressor.net.params['fc7-newb'][1].data

model.conv1.weight = torch.nn.Parameter(torch.FloatTensor(conv1_w), requires_grad=False)
model.conv2.weight = torch.nn.Parameter(torch.FloatTensor(conv2_w), requires_grad=False)
model.conv3.weight = torch.nn.Parameter(torch.FloatTensor(conv3_w), requires_grad=False)
model.conv4.weight = torch.nn.Parameter(torch.FloatTensor(conv4_w), requires_grad=False)
model.conv5.weight = torch.nn.Parameter(torch.FloatTensor(conv5_w), requires_grad=False)
model.fc6.weight = torch.nn.Parameter(torch.FloatTensor(fc6_w), requires_grad=False)
model.fc7.weight = torch.nn.Parameter(torch.FloatTensor(fc7_w), requires_grad=False)
model.fc7b.weight = torch.nn.Parameter(torch.FloatTensor(fc7b_w), requires_grad=False)

model.conv1.bias = torch.nn.Parameter(torch.FloatTensor(conv1_b), requires_grad=False)
model.conv2.bias = torch.nn.Parameter(torch.FloatTensor(conv2_b), requires_grad=False)
model.conv3.bias = torch.nn.Parameter(torch.FloatTensor(conv3_b), requires_grad=False)
model.conv4.bias = torch.nn.Parameter(torch.FloatTensor(conv4_b), requires_grad=False)
model.conv5.bias = torch.nn.Parameter(torch.FloatTensor(conv5_b), requires_grad=False)
model.fc6.bias = torch.nn.Parameter(torch.FloatTensor(fc6_b), requires_grad=False)
model.fc7.bias = torch.nn.Parameter(torch.FloatTensor(fc7_b), requires_grad=False)
model.fc7b.bias = torch.nn.Parameter(torch.FloatTensor(fc7b_b), requires_grad=False)

with open('tracker.init.pt', 'w') as f:
	torch.save(model.state_dict(), f)

