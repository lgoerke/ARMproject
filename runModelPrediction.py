# Import Keras and VGG16 - Neural Network Model
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import id_to_synset

# Load an image (Use more later)
categories = ["table", "car", "airplane", "church", "fruit", "house", "dog", "cat", "teapot", "table lamp", "castle", "pillow", "volcano", "coffee mug", "envelope"]
categoryPaths = [[category + '/0000000' + str(i) + '.jpg' for i in range(500)] for category in categories]

for paths in categoryPaths[0]:

	# And resize it to fit
	imgs = preprocess_image_batch([path[0:10]],img_size=(256,256), crop_size=(224,224), color_mode="bgr")

	# Specify Model Parameters, load pretrained weights and compile Model
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model = convnet('vgg_16',weights_path="vgg16_weights.h5", heatmap=False)
	model.compile(optimizer=sgd, loss='mse')

	# Predict image 
	out = model.predict(imgs)

# Index of maximum gives us the Synset = Category
#print(id_to_synset(out.argmax()))

# Maximum value corresponds to the networks "certainty" 
#print(out.max())
print(out)