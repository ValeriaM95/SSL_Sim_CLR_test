# Thesis

#Self-supervised learning for scalp images recognition

SimCLR

Steps:

1. Create 2 versions of an image by applying data augmentation techniques

2. Apply a CNN (like ResNet) and obtain as output a 1D feature vector

3. Apply a small MLP on that feature vector

4. The output features of the two augmented images are then trained to be close to each other, while all other images in that batch should be as different as possible.


![12](https://user-images.githubusercontent.com/108786717/190957085-7b6a355f-4c74-443a-9798-8cfa078cee15.PNG)

Official paper: https://arxiv.org/pdf/2002.05709v3.pdf 
![12](https://user-images.githubusercontent.com/108786717/190957238-e7f6e079-7907-4fe1-aef5-5c355667d2c8.PNG)


Files description:

loss.py - custom Loss function

training_function - function SimCLR

experiment.ipynb - 2 models (fully supervised, pretext task)
