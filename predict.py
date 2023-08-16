import torch
from torchvision.models import get_model
from PIL import Image
import torchvision.transforms as T
import argparse
from utils.augment import create_AugTransforms, list_augments

def parse_opt():
    parsers = argparse.ArgumentParser()

    parsers.add_argument('--img', default='./img.jpg', type=str)
    parsers.add_argument('--pt', default='./best.pt', type=str)
    parsers.add_argument('--num_classes', default=7, type=int)
    parsers.add_argument('--transforms', default='centercrop_resize to_tensor_without_div')
    parsers.add_argument('--imgsz', default='[[720, 720], [224, 224]]', type=str)

    args = parsers.parse_args()
    return args

def image_process(path: str, transforms: T.Compose):
    img = Image.open(path).convert('RGB')
    return transforms(img).unsqueeze(0)

def main(opt):

    # variable
    img_path = opt.img
    weight_path = opt.pt
    transforms = opt.transforms
    imgsz = opt.imgsz
    num_classes = opt.num_classes

    # image
    image = image_process(img_path, create_AugTransforms(transforms, eval(imgsz)))

    # model centercrop_resize 
    model = get_model('mobilenet_v2', width_mult = 0.25)
    
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    weight = torch.load(weight_path, map_location='cpu')['model']
    model.load_state_dict(weight)
    # eval
    model.eval()

    out = model(image)
    print(torch.nn.functional.softmax(out, dim=-1))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)