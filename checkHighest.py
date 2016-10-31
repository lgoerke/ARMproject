import numpy as np 
import json
from convnetskeras.imagenet_tool import id_to_words
from scipy.io import savemat

indices = {'table':[526, 736, 532],'car':[407, 436, 468, 511, 609, 627, 656, 661, 751, 817],"airplane":[404], "church":[497], "fruit":[988, 989, 998, 952, 953, 954, 955, 956, 957, 948, 949, 950, 951, 990, 984, 987, 948], "house":[698, 663], "dog":[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156], "teapot":[849], "castle":[483], "volcano":[980], "coffee mug":[504]}

with open('MeasuredData(questionnaire+network)/shuffled1.txt','r') as f:
	f.readline()
	f.readline()
	f.readline()
	images = f.readline()


obj_arr = np.zeros((110,3), dtype=np.object)
i = 0

cats = {}
for key in indices.keys():

	with open('classifications/' + key + '.json','r') as f:
		imgs = json.load(f)
		cats[key] = imgs

for element in images[1:-2].split(', '):

	str = element.split('_')
	print(element)
	key = str[1][:-1]
	indx = int(str[0][1:])

	imgs = cats[key]
	
	im = np.array(imgs[key][indx])
	obj_arr[i][0] = indx
	obj_arr[i][1] = key
	obj_arr[i][2] = np.sum(im[indices[key]])
	i +=1		


		#max = np.argmax(imgs[key][indx])
		#if key == 'airplane':
			#print(imgs[key][indx][404])
		#	
		#	print(indx, np.sum(im[indices[key]]))
		#if max not in indices[key]:
		#	print(key, indx, id_to_words(max), max)
savemat('networkClassification.mat', mdict={'classifications': obj_arr})