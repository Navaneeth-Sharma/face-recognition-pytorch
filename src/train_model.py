import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN
from utils.dataloader import TripletFolder
import torchvision.transforms as transforms
from config import TRAIN_FOLDDER_PATH, TRAIN_BATCH_SIZE
from utils.loss import TripletLoss
from models.resnet34_128emb import SiameseNetwork


def train(save_path="../models/resnet34_siamesenet.pth"):

    mtcnn = MTCNN(image_size=224, margin=50)
    siamese_dataset = TripletFolder(
        TRAIN_FOLDDER_PATH,
        transform=transforms.Compose(
            [mtcnn, transforms.ColorJitter(brightness=[0.8, 1.5], contrast=[1, 2],),]
        ),
    )

    train_dataloader = DataLoader(
        siamese_dataset, shuffle=True, num_workers=2, batch_size=TRAIN_BATCH_SIZE
    )

    model = SiameseNetwork().cuda()
    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    loss_history = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    numepochs = 30

    for epoch in tqdm(range(numepochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img) in enumerate(
            tqdm(train_dataloader, desc="Training", leave=False)
        ):

            anchor_img = anchor_img.to(device)  # send image to GPU
            positive_img = positive_img.to(device)  # send image to GPU
            negative_img = negative_img.to(device)  # send image to GPU

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())

        loss_history.append(np.mean(running_loss))
        print(
            "Epoch: {}/{} - Loss: {:.4f}".format(
                epoch + 1, numepochs, np.mean(running_loss)
            )
        )

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
