import unittest
import sys
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms as T
from torch.autograd import Variable
from PIL import Image
sys.path.append('..')
from src.models.resnet34_128emb import SiameseNetwork


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


class siameseNetworkTest(unittest.TestCase):

    def test_positive(self):
        self.assertEquals(
            predict('../references/items/rdj.png', '../references/items/rdj_2.png')[1],
            True
        )

    def test_negative(self):
        self.assertEquals(
            predict('../references/items/rdj.png', '../references/items/tom_holand.png',
            sim_threshold=0.95)[1],
            False
        )

if __name__ == "__main__":
    unittest.main()
