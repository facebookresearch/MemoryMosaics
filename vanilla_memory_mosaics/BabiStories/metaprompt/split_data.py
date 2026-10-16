# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

import json 
from collections import Counter
from tqdm import tqdm 
import math 
verb = []
noun = []
adj  = []
feature = []
from tqdm import tqdm 
for k in tqdm(range(25)):
	if k < 10:
		mark = '0%d'  % k 
	else:
		mark = '%d' % k 

	with open(f'data{mark}.json', 'r') as f:
		data = json.load(f)
	
	length = len(data)
	split_size = 1000
	
	for i in range(math.ceil(length / split_size)):	
		f = open(f"data{mark}_{i}.json", "w") 
		json.dump(data[i*split_size:(i+1)*split_size], f)
		f.close() 

