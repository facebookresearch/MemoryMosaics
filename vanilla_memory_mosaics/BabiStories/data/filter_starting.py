# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.



startings = {1:[], 2:[], 3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[]}
for name in ['','p2']:

	for i in range(128):
		f = open(f'startings{i}{name}.txt', 'r')
		data = f.readlines()
		lines = []
		for line in data:
			line = line.replace('...','')
			split = line.find('. ')

			if split == -1:
				continue
			line = line[split+2:].replace('.','')
			line = line.strip().split(' ')
			for start_length in [1,2,3,4,5,6,7,8,9,10,11,12]:
				if len(line) < start_length:
					break
				per_start = ' '.join(line[:start_length]).strip() + '\n'
				startings[start_length].append(per_start)
				#print(per_start)

f = open('allstartings.txt','w')
total_num = 0
for key in startings:
	startings[key] = list(set(startings[key]))
	total_num += len(startings[key])
	print(len(startings[key]))
	f.writelines(startings[key])
f.close()



