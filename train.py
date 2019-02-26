import torch
import torch.nn.functional as Functional
from torchvision import datasets, models, transforms
import numpy as np
import argparse
import sys
import os
import util
from util import (
    train_model,
    load_datasets,load_categories,
    device_type)
from collections import OrderedDict

# Command Line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store' , default="./flowers/")
    parser.add_argument('--save_dir', dest="save_dir",default="./checkpoint.pth" )
    parser.add_argument('--arch', dest="arch", default="vgg16", type=str)
    parser.add_argument('--learning_rate', dest="learning_rate", type=float, default=0.01)
    parser.add_argument('--hidden_units', dest="hidden_units", type=int, default=1000)
    parser.add_argument('--epochs', dest="epochs", type=int, default=10)
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    image_datasets = load_datasets(args.data_dir)
    device = device_type(args.gpu)
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
   # if args.arch == "vgg16":
        num_ftrs = model.classifier[0].in_features
        classifier = torch.nn.Sequential(OrderedDict([
                          ('fc1', torch.nn.Linear(num_ftrs, 4096)),
                          ('relu', torch.nn.ReLU()),
                          ('dropout', torch.nn.Dropout(p=0.5)),
                          ('fc2', torch.nn.Linear(4096, args.hidden_units)),
                          ('relu', torch.nn.ReLU()),
                          ('dropout', torch.nn.Dropout(p=0.5)),
                          ('fc3', torch.nn.Linear(args.hidden_units, 102))]))
    
        model.classifier = classifier
        optimizer = torch.optim.SGD(model.classifier.parameters(),
                                    lr=args.learning_rate,
                                    momentum=0.9)
  #  else:
    
    #num_ftrs = model.fc.in_features
     #   model.fc = torch.nn.Linear(num_ftrs, 102)
      #  optimizer = torch.optim.SGD(model.fc.parameters(),
              
        #lr=args.learning_rate,
         #                           momentum=0.9)
        
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=40, 
                                                gamma=0.1)
    model = model.to(device)
    model = train_model(model, criterion, optimizer, scheduler, 
                        image_datasets, args.gpu, epochs=10)
    model.class_to_idx = image_datasets['train'].class_to_idx
    model_name = "checkpoint2.pth"
    save_dir = args.save_dir or ""
    checkpoint = {"arch": args.arch , 
                  "optimizer": optimizer,
                  "state_dict": model.state_dict(),
                  "class_to_idx": model.class_to_idx,
                  "hidden_units": args.hidden_units,
                  "learning_rate": args.learning_rate,
                  "classifier":model.classifier,
                  "epochs": args.epochs}
    
    
    
if __name__ == "__main__":
    main()
    
print("Training completed") 
