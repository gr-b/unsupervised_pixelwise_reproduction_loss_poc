# nsupervised Pixelwise ReproductionLoss Proof of Concept
This is a toy example for using unsupervised regression to learn about an image.
In this example, the model learns the location of the center of an image (image is a generated circular gradient).
The model learns center coordinates by attempting to reconstruct the original image. The distance between the original 
image and the reconstruction is the loss.
