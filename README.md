# Face Verification with Mask

Face Verification is a complex problem and is evolving every year. There are many challenges one can face while creating the Face Verifcation System. To solve this problem there is a popular approach of One Shot learning.

**WORK UNDER PROGRESS**

## Demo of the Project

https://user-images.githubusercontent.com/63489382/163522027-cf2475d4-b0a9-4f73-9d7c-8fdbe38c3441.mp4


## Block Diagram

![](references/items/Blank%20diagram.png)

## Architecture

The is a network which has shared weights b/w all the data. There are multiple architectures to build this network. Trying with multiple networks the **Resnet34** based architecture gives the good result. The training accuracy is about 85% and validation accuracy is about 79%.

![](references/items/Untitled%20Diagram.drawio.png)

## Loss Function Used

The Triplet Loss function is used to achive the results.
