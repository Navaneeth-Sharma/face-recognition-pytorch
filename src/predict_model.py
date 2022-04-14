import torch
import argparse
from facenet_pytorch import MTCNN
from torchvision import transforms as T
from models.resnet34_128emb import SiameseNetwork
from torch.autograd import Variable
from PIL import Image


def predict(anchor_img_path: str, test_img_path: str, sim_threshold=0.9):
    """
    This function takes mainly 2 arguments.
    anchor_img_path: The image path of the person 1
    test_img_path: The image path of the person 2

    returns: Bool (True if 2 people are same else False)
    """

    MODEL_PATH = "../models/resnet34_siamesenet.pth"
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNetwork().to(device=map_location)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=map_location))
    model.eval()

    tfms = T.Compose([MTCNN(image_size=224, margin=50)])

    anchor_img = Image.open(anchor_img_path).convert("RGB")
    anchor_img = Variable(tfms(anchor_img)).to(device=map_location)
    emb0 = model(anchor_img.unsqueeze(0))

    test_img = Image.open(test_img_path).convert("RGB")
    test_img = Variable(tfms(test_img)).to(device=map_location)
    emb1 = model(test_img.unsqueeze(0))

    sim = torch.cosine_similarity(emb0, emb1)

    if sim > sim_threshold:
        return sim, True
    return sim, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict whether 2 images are similar or not"
    )
    parser.add_argument("-aimg", "--anchor_img", nargs="?", default="")
    parser.add_argument("-timg", "--test_img", nargs="?", default="")
    parser.add_argument("-st", "--sim_threshold", nargs="?", default=0.9, type=float)
    args = parser.parse_args()

    result = predict(
        anchor_img_path=args.anchor_img,
        test_img_path=args.test_img,
        sim_threshold=args.sim_threshold,
    )

    if result[1]:
        print(
            "The images are of same people!! with the\
         similarity score:",
            result[0].detach().numpy()[0],
        )
    else:
        print(
            "The images are of different people!! with the\
         similarity score:",
            result[0].detach().numpy()[0],
        )
