import pickle

categories = ["table", "car", "airplane", "church", "fruit", "house", "dog", "teapot", "castle", "volcano", "coffee mug"]

pkl_file = open('n_prob_list2.pkl', 'rb')
x = pickle.load(pkl_file)
for cat in categories:
	p = []
	print(cat)
	for i in range(len(x)):
		if x[i][1] == cat:
			p.append(x[i][2])
	p.sort()
	for i in range(len(p)):
		print(p[i])
	print('---')