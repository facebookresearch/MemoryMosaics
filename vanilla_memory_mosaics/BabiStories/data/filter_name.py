# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

names = []
for i in range(128):
	f = open(f'names{i}.txt', 'r')
	data = f.readlines()
	lines = []
	for line in data:
		split = line.find('. ')
		if split == -1:
			continue
		name = line[split+2:].strip()+'\n'
		if len(name) > 8:
			continue
		names.append(name)

print(len(names))
names = list(set(names))
print(len(names))

f = open('allnames.txt','w')

f.writelines(names)
f.close()
