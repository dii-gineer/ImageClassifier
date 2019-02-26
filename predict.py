import argparse
import torch
import torch.nn.functional as Functional
from torchvision import transforms, models
import json
from PIL import Image
import numpy as np
from util import (
    load_categories,
    device_type)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store', default='./flowers/test/1/image_06752.jpg', nargs='*', type = str)
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
    parser.add_argument('--top_k', dest='top_k', type=int, default=5)
    parser.add_argument('--category_names', dest="category_names", default="cat_to_name.json")
    parser.add_argument('--gpu', action="store_true", default=False)
    return parser.parse_args()



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image =  Image.open(image)
  
      # TODO: Process a PIL image for use in a PyTorch model
    current_width, current_height = image.size 
    if current_width<current_height:
        new_height = int(current_height*256/current_width)
        image=image.resize((256,new_height))
    else:
        new_width = int(current_width*256/current_height)
        image= image.resize((new_width,256))
        
     #Crop
    
    precrop_width, precrop_height = image.size
    
    left = int((precrop_width - 224)/2)
    top= int((precrop_height - 224)/2)
    right = int((precrop_width + 224)/2)
    bottom = int((precrop_height + 224)/2)
    
    cropped_image = image.crop((left,top,right,bottom))
    
    #normalize 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = np.array(cropped_image) / 255
    image_norm = (image_array - mean ) / std
    
    img_transpose =image_norm.transpose((2,0,1))  

    img_tensor=torch.from_numpy(img_transpose)
    return  img_tensor 
    


def predict(image_path, model, device,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    output = model(processed_image.float().to(device))
    probs, indices = torch.topk(Functional.softmax(output, dim=1), topk, sorted=True)
    idx_to_class = { v:k for k, v in model.class_to_idx.items()}
    return [prob.item() for prob in probs[0].data], [idx_to_class[ix.item()] for ix in indices[0].data]

def load_checkpoint(path):
    checkpoint= torch.load(path)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def main():
    args = parse_args()
    device = device_type(args.gpu)
    model = load_checkpoint(args.checkpoint)
    model = model.to(device)
    image = args.input
    cat_to_name = load_categories(args.category_names)   
    probs, classes = predict(image, model, device, args.top_k)
    print(probs, [cat_to_name[name] for name in classes])

    
if __name__ == "__main__":
    main()

