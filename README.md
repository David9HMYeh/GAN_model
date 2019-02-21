# GAN Model
Anomaly detection usage in 32*32 pixel.
----------------
Create convolution layers share by D and G as Frontend.<br>
D is the discriminator for identifying the real picture or fake picture.<br>
G is generater which sample noise from uniform distribution and try to match with the latent variable.<br>

Use mu and mean from convolution layers to generate images.<br>
The loss function is combine reconstruction error, generater error, and discriminator error.<br>
D learning rate has to be higher than G for better result. With small learning rate, G is able to generate realistic image<br>

ProductionLine_imgae_detection.ipynb file show the trained model applied on real manufatureing image and able to identified anomaly image using cross entorpy with accuracy 96%.

model_result.png show the sample image from generated model compare with real image.
