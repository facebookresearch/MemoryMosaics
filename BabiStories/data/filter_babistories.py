# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm
import json 
import os 

stories = []
for k in tqdm(range(23)):
	mark = '0%d'  % k if k < 10 else  '%d' % k 
	for j in range(100):
		alldata = []
		if not os.path.exists(f'{mark}_{j}.txt'):
			print(f'{mark}_{j}.txt')
			continue
		with open(f'{mark}_{j}.txt', 'r') as f:
			data = f.readlines()
		for i in (range(len(data))):
			story = json.loads(data[i])
			
			# elimite prompt
			story = story[story.find("[/INST]")+7:].strip()

			# elimite unrelated endings
			split_symbs = ["This is the moral of the story:", "The story ends with a moral:","They knew the moral:","(Moral of the story", "However, the moral of the story is this:","It's the moral of our story","The story teaches us a moral:",\
			"The moral value:", "Here is a moral:","So the moral is:"," Moral of the Story:","That is the moral of the story","The moral of this", "And that's the moral of", "This story has a moral:","They learned a moral:","Here's your little moral:","And so, they all learned the moral:",\
			" And the moral is:","And that's the moral value of", "So, the moral of","This story has a moral value:","That was the moral of the story", "In this story, the moral is","And that is the moral of this story",\
			 "Here is the moral of our story","This is a good moral for","The moral value here is that","And that's the end, with a very important moral","The hidden moral:","So the moral of this story is:","And so, the moral of the story is that","That's the moral of the story.",\
			"This is the moral","And that is the moral of the story","And that was the moral of","And what's the moral of this story?","And so, the moral of the story is:", "**Moral of the Story:**", "The moral was:", "conflict:","bad ending:","he moral lesson is:","The moral of our story is",\
			 "**Moral of the story:**", "**Moral Value:", "The moral of the story","The moral is", "The moral:", "*Moral:", "The moral value of this story", "So, the moral is", \
			 "So the moral of the story", " And that is the moral of the story.", "Moral of the story:","*Moral*:","**Moral**:","* Moral Value:","* Unexpected Event:","* Plot Twist:","Moral Value:","That's the moral:","**Moral value:","* Moral value:","And the moral of","Moral:", "The moral?", \
			"Story key words:","Story Keywords:","Theme:","> [Author's Note]", "Confused by the bad words?", "* Pretty = beautiful", "\\n\n> [Author's Note]", "Words:",\
			"(Note to the moderators", "(3-5 simple paragraphs)", "Confused?","Plot twist:","Dialogue:" "Confused by the bad words?", "Plot twist:", "Conflict:","Use of words:", "List of simple words used:","Here are some of the simple words used",
			'(Note:', '*Note','*\n\n*Note:', "Note:", "##### *Note:", "###\n\n(Note:", "*(Note:","* Note :","Notes:","(Note to parents",\
					 "The end", "The end!", "Note to Parents:","Notes:",'The End', "Characters:", "Development Notes:", "Confused by the bad words?", "** Note on difficulty", "* (Note to reader:", "THE END", '##', "###", "####"]
			
			for symb in split_symbs:
				split = story.find(symb)	
				if split != -1:
					story = story[:split]
			story = story.strip().replace('\n\n', ' ')
			

			user_symbs = [f'User {i}:' for i in range(10)]
			for symb in user_symbs:
				split = story.find(symb)	
				if split != -1:
					story = story[:split]
			story = story.strip().replace('\n\n', ' ').strip()

			if story.find("https:")>-1 or story.find("http") > -1 or story.find('==') > -1 or story.find('---') > -1 or \
					story.find('\\')>-1 or story.find('*') > -1 or story.find('#') > -1 or story.find("<") > -1 or story.find(">") > -1 or \
					story.find("&quot") > -1 or story.find('&nbsp') > -1 or story.find("%") > -1 or story.find("^") > -1 or story.find("/") > -1 or story.find("@") > -1 or \
					story.find("[") > -1 or story.find("]") > -1 or story.find("{") > -1 or story.find("}") > -1 or story.find("+") > -1:

				continue
			if story.count('(') >0 or story.count(')') >0:
				continue 


			stories.append(json.dumps(story)+'\n')
print(len(stories))
f = open(f'stories_clean.txt', 'w')
f.writelines(stories)
f.close()


