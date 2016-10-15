import json
from convnetskeras.imagenet_tool import id_to_synset
from random import shuffle
from shutil import copyfile
import pickle
import numpy, scipy.io

#categories = ['table']
#categories = ["table", "car", "airplane", "church", "fruit", "house", "dog", "cat", "teapot", "table lamp", "castle", "pillow", "volcano", "coffee mug", "envelope"]
categories = ["table", "car", "airplane", "church", "fruit", "house", "dog", "teapot", "castle", "volcano", "coffee mug"]
#indices = {'table':[526, 736, 532],'car':[407, 436, 468, 511, 609, 627, 656, 661, 751, 817],"airplane":[404], "church":[497], "fruit":[988, 989, 998, 952, 953, 954, 955, 956, 957, 948, 949, 950, 951, 990, 984, 987, 948], "house":[698, 663], "dog":[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156], "cat":[285, 283, 282, 284, 281], "teapot":[849], "table lamp":[846], "castle":[483], "pillow":[721], "volcano":[980], "coffee mug":[504], "envelope":[549]}
indices = {'table':[526, 736, 532],'car':[407, 436, 468, 511, 609, 627, 656, 661, 751, 817],"airplane":[404], "church":[497], "fruit":[988, 989, 998, 952, 953, 954, 955, 956, 957, 948, 949, 950, 951, 990, 984, 987, 948], "house":[698, 663], "dog":[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156], "teapot":[849], "castle":[483], "volcano":[980], "coffee mug":[504]}
steps = [0,55,110,165,220,279,334,389,444,499]

cat_classifications = {'table':[1, 2, 3]}
for category in categories:
	data = json.load(open('classifications/' + category + '.json'))
	data = data[category]
	classifications = []
	for img in data:
		# add up relevant values
		res = 0
		for index in indices[category]:
			res = res + img[index]
		classifications.append(res)
	cat_classifications[category] = classifications

chosen_indices = {'table':[1,2,3]}
prob_cat = {'table':[1,2,3]}
for category in categories:
	# sort list to get imgs of a spectrum of percentages
	indices_sum = [i[0] for i in sorted(enumerate(cat_classifications[category]), key=lambda x:x[1])]
	chosen_imgs = [indices_sum[step] for step in steps]
	probabilities = [cat_classifications[category][i] for i in chosen_imgs]
	prob_cat[category]=probabilities
	chosen_indices[category] = chosen_imgs

pickle.dump(chosen_indices, open('chosen_indices1.pkl','wb'))
print(chosen_indices)
#print(prob_cat)
selection = []
for category in categories:
	indices = chosen_indices[category]
	for ind,i in enumerate(indices):
		s = str(i) + '_' + category
		selection.append(s)

shuffle(selection)
pickle.dump(selection, open('selection1.pkl','wb'))
print(selection)

prob_list=[]
for ind,img in enumerate(selection):
	tmp = img.split('_')
	filename = 'imgs/' + tmp[1] + '/0000000' + tmp[0] + '.jpg'
	copyfile(filename, 'imgs/shuffled1/' + str(1000+ind) + '_' + tmp[1] + '.jpg')
	var = cat_classifications[tmp[1]]
	index = chosen_indices[tmp[1]]
	var = index.index(int(tmp[0]))
	prob = cat_classifications[category][var]
	prob_list.append(numpy.array([1000+ind,tmp[1],prob]))

print(prob_list)
pickle.dump(prob_list, open('prob_list1.pkl','wb'))
scipy.io.savemat('prob_list1.mat', mdict={'probability_list1':numpy.array(prob_list,dtype=numpy.object)})


