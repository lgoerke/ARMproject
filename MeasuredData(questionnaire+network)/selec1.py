import pickle

selection = ['380_teapot', '87_car', '454_fruit', '80_car', '329_dog', '64_table', '237_castle', '300_church', '447_volcano', '54_church', '476_church', '269_castle', '139_dog', '3_house', '160_dog', '337_airplane', '231_coffee mug', '46_teapot', '443_volcano', '365_coffee mug', '99_table', '243_fruit', '204_volcano', '428_car', '105_fruit', '257_church', '465_coffee mug', '200_dog', '437_house', '249_airplane', '233_church', '295_teapot', '488_coffee mug', '144_house', '334_volcano', '252_volcano', '356_teapot', '144_fruit', '259_teapot', '184_table', '483_castle', '172_castle', '109_church', '245_church', '1_house', '285_coffee mug', '401_dog', '143_table', '419_church', '244_fruit', '415_volcano', '366_fruit', '315_castle', '407_airplane', '318_airplane', '369_house', '370_house', '2_dog', '193_table', '461_car', '307_airplane', '359_car', '196_car', '418_airplane', '352_car', '74_volcano', '142_dog', '310_fruit', '304_car', '379_fruit', '9_coffee mug', '219_castle', '188_airplane', '150_table', '178_volcano', '234_castle', '130_teapot', '12_teapot', '65_coffee mug', '311_volcano', '201_church', '29_table', '200_teapot', '192_coffee mug', '465_house', '64_coffee mug', '492_dog', '298_volcano', '263_table', '476_car', '239_airplane', '378_car', '479_table', '406_table', '243_castle', '269_church', '6_coffee mug', '62_house', '168_teapot', '117_dog', '308_castle', '233_house', '331_teapot', '162_airplane', '298_fruit', '497_dog', '83_castle', '386_fruit', '124_house', '321_airplane']
pickle.dump(selection, open('selection1a.pkl','wb'))