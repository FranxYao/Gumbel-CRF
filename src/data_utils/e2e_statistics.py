"""Test the E2E dataset"""


import csv
import numpy as np 

from collections import Counter
from pprint import pprint

e2e_path = {'train': '../data/e2e-dataset/trainset.csv',
            'dev': '../data/e2e-dataset/devset.csv',
            'test': '../data/e2e-dataset/testset.csv'}
output_path = '../outputs/'

with open(e2e_path['train']) as fd:
  reader = csv.reader(fd)
  lines = [l for l in reader]
  lines = lines[1:]

## tag statistics 

tags = []
for l in lines:
  t = l[0].split(', ')
  for ti in t:
    tags.append(ti.split('[')[0])
tags = Counter(tags)
pprint(tags)
# >> 
# Counter({'name': 42061,
#          'food': 35126,
#          'priceRange': 29127,
#          'customer rating': 28090,
#          'familyFriendly': 26295,
#          'area': 24716,
#          'near': 20546,
#          'eatType': 20111})

## Sentence templates extraction -- can be used for supervised learning 
sents = [l[1] for l in lines]
print(sents[:5])
# >> 
# ['The Vaults pub near Café Adriatic has a 5 star rating.  Prices start at £30.',
#  'Close to Café Brazil, The Cambridge Blue pub serves delicious Tuscan Beef '
#  'for the cheap price of £10.50. Delicious Pub food.',
#  'The Eagle is a low rated coffee shop near Burger King and the riverside that '
#  'is family friendly and is less than £20 for Japanese food.',
#  'Located near The Sorrento is a French Theme eatery and coffee shop called '
#  'The Mill, with a price range at £20-£25 it is in the riverside area.',
#  'For luxurious French food, the Loch Fyne is located by the river next to The '
#  'Rice Boat.']

# replace v with k 
sent_temp = []
for l in lines:
  kv = l[0].split(', ')
  k = [kv_.split('[')[0] for kv_ in kv]
  v = [kv_.split('[')[1].split(']')[0] for kv_ in kv]
  s = l[1].replace('.', ' .')
  for k_, v_ in zip(k, v):
    if(v_ in s): s = s.replace(v_, '_' + k_.replace(' ', '_') + '_')
  sent_temp.append(s)
sent_temp_cnt = Counter(sent_temp)
pprint(sent_temp_cnt.most_common(5))
# >> 
# [('_name_ provides _food_ food in the _priceRange_ price range. It is located '
#   'in the _area_. It is near _near_. Its customer rating is _customer_rating_.',
#   81),
#  ('_name_ provides _food_ food in the _priceRange_ price range. It is located '
#   'in the _area_. Its customer rating is _customer_rating_.',
#   71),
#  ('_name_ provides _food_ food in the _priceRange_ price range. Its customer '
#   'rating is _customer_rating_.',
#   54),
#  ('_name_ is a _eatType_ providing _food_ food in the _priceRange_ price '
#   'range. It is located in the _area_. It is near _near_. Its customer rating '
#   'is _customer_rating_.',
#   45),
#  ('_name_ provides _food_ food in the _priceRange_ price range. It is located '
#   'in the _area_.',
#   44)]
temp_occ = np.array(list(sent_temp_cnt.values()))
print(temp_occ[np.where(temp_occ > 1)].sum())
# >> 2990
print(temp_occ.sum())
# >> 42061

with open(output_path + 'e2e_temp.txt', 'w') as fd:
  for k, v in sent_temp_cnt.most_common():
    fd.write('%s\t\t%d\n' % (k, v))

# Conclusion: templates are sparce, and context-dependent (not surprising)


## condense templates 
sent_temp_condensed = []
for s in sent_temp:
  s = s.split()
  s_ = []
  for w in s:
    if(w[0] == '_'): s_.append(w)
    else: 
      if(len(s_) == 0): s_.append('CHUNK')
      if(len(s_) > 0 and s_[-1] != 'CHUNK'): s_.append('CHUNK')
  sent_temp_condensed.append(' '.join(s_))
sent_temp_condensed_cnt = Counter(sent_temp_condensed)

pprint(sent_temp_condensed_cnt.most_common(5))
# >> 
# [('_name_ CHUNK', 1036),
#  ('_name_ CHUNK _near_ CHUNK', 753),
#  ('_name_ CHUNK _food_ CHUNK', 656),
#  ('_name_ CHUNK _eatType_ CHUNK', 613),
#  ('_name_ CHUNK _eatType_ CHUNK _near_ CHUNK', 438)]
# from the above case it is very clear why Wiseman (18) use he HSMM model 

temp_occ = np.array(list(sent_temp_condensed_cnt.values()))
print(np.sum(temp_occ > 5))
# >> 882
print(np.sum(temp_occ > 1))
# >> 3813
print(temp_occ[np.where(temp_occ > 1)].sum())
# >> 31058
print(temp_occ.sum())
# >> 42061

with open(output_path + 'e2e_condensed_temp.txt', 'w') as fd:
  for k, v in sent_temp_condensed_cnt.most_common():
    fd.write('%s\t\t%d\n' % (k, v))