import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
    
from lib.averageMeter import AverageMeters
from lib.logger import colorlogger
from lib.timer import Timers
from lib.averageMeter import AverageMeters
from lib.torch_utils import adjust_learning_rate

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from test import test
import time

NAME = "release_base"

# Set `LOG_DIR` and `SNAPSHOT_DIR`
def setup_logdir():
    timestamp = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) 
    LOGDIR = os.path.join("logs", "%s_%s"%(NAME, timestamp))
    SNAPSHOTDIR = os.path.join("snapshot", "%s_%s"%(NAME, timestamp))
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
    if not os.path.exists(SNAPSHOTDIR):
        os.makedirs(SNAPSHOTDIR)
    return LOGDIR, SNAPSHOTDIR
LOGDIR, SNAPSHOTDIR = setup_logdir()

# Set logging 
logger = colorlogger(log_dir=LOGDIR, log_name="train_logs.txt")

# Set Global Timer
timers = Timers()

# Set Global AverageMeter
averMeters = AverageMeters()
    
def train(model, dataloader, optimizer, epoch, iteration):
    # switch to train mode
    model.train()
    
    averMeters.clear()
    end = time.time()
    for i, inputs in enumerate(dataloader): 
        averMeters["data_time"].update(time.time() - end)
        iteration += 1
        
        lr = adjust_learning_rate(optimizer, iteration, BASE_LR=0.0002,
                         WARM_UP_FACTOR=1.0/3, WARM_UP_ITERS=1000,
                         STEPS=(0, 14150*15, 14150*20), GAMMA=0.1)  
        
        # forward
        outputs = model(**inputs)
        
        # loss
        loss = outputs
            
        # backward
        averMeters["loss"].update(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        averMeters["batch_time"].update(time.time() - end)
        end = time.time()
        
        if i % 10 == 0:
            logger.info("Epoch: [{0}][{1}/{2}]\t"
                  "Lr: [{3}]\t"
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  "loss {loss.val:.5f} ({loss.avg:.5f})\t"
                  .format(
                      epoch, i, len(dataloader), lr, 
                      batch_time=averMeters["batch_time"], data_time=averMeters["data_time"],
                      loss=averMeters["loss"])
                 )
        
        if i % 5000 == 0:  
            torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, "%d_%d.pkl"%(epoch,i)))
            torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, "last.pkl"))
        
    return iteration

class Dataset():
    def __init__(self,test=True):
        if test:
            ImageRoot = "./data/coco2017/val2017"
            AnnoFile = "./data/coco2017/annotations/person_keypoints_val2017_pose2seg.json"
        else:
            ImageRoot = "./data/coco2017/train2017"
            AnnoFile = "./data/coco2017/annotations/person_keypoints_train2017_pose2seg.json"     
        self.datainfos = CocoDatasetInfo(ImageRoot, AnnoFile, onlyperson=True, loadimg=True)
    
    def __len__(self):
        return len(self.datainfos)
    
    def __getitem__(self, idx):
        rawdata = self.datainfos[idx]
        img = rawdata["data"]
        image_id = rawdata["id"]
        height, width = img.shape[0:2]
        gt_kpts = np.float32(rawdata["gt_keypoints"]).transpose(0, 2, 1) # (N, 17, 3)
        gt_segms = rawdata["segms"]
        gt_masks = np.array([annToMask(segm, height, width) for segm in gt_segms])
    
        return {"img": img, "kpts": gt_kpts, "masks": gt_masks}
        
    def collate_fn(self, batch):
        batchimgs = [data["img"] for data in batch]
        batchkpts = [data["kpts"] for data in batch]
        batchmasks = [data["masks"] for data in batch]
        return {"batchimgs": batchimgs, "batchkpts": batchkpts, "batchmasks":batchmasks}
        
if __name__=="__main__":
    logger.info("===========> loading model <===========")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Pose2Seg Training")
    parser.add_argument(
        "--weights",
        help="path to .pkl model weight",
        type=str,
        default=None
    )
    args = parser.parse_args()
    model = Pose2Seg().to(device)
    if args.weights:
        model.init(args.weights)
    model.train()
   
    
    logger.info("===========> loading data <===========")
    datasetTrain = Dataset(test=False)
    #modify it later
    torch.manual_seed(3407)
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=4, shuffle=True,
                                                   num_workers=4, pin_memory=False,
                                                   collate_fn=datasetTrain.collate_fn)


    logger.info("===========> set optimizer <===========")
    ''' set your optimizer like this. Normally is Adam/SGD. '''
    #optimizer = torch.optim.SGD(model.parameters(), 0.0002, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001, weight_decay=0.005)
    iteration = 0
    epoch = 0
    max = 25
    try:
        while iteration < 14150*max:
            logger.info("===========>   training    <===========")
            iteration = train(model, dataloaderTrain, optimizer, epoch, iteration)
            epoch += 1
            
            logger.info("===========>   testing    <===========")
            # test(model, dataset="cocoVal", logger=logger.info)
            test(model, dataset="OCHumanVal", logger=logger.info)
            time.sleep(60) 



    except (KeyboardInterrupt):
        logger.info("Save ckpt on exception ...")
        torch.save(model.state_dict(), os.path.join(SNAPSHOTDIR, "interrupt_%d_%d.pkl"%(epoch,iteration)))
        logger.info("Save ckpt done.")
#TODO: get max IOU result out 