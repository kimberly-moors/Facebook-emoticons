def emoji_featurizer (dataset):
	#the emoji's placed into a dictionary.
	emojidictionary = {}
	emojidictionary['happy'] = [':)',':]','XD',':-)','^_^',':D',':P','xD',':}',':>', '=)']
	emojidictionary['random'] = ['<3',':$',':o']
	emojidictionary['winking'] = [';)',';-)']
	emojidictionary['neutral'] = [':|','-.-']
	emojidictionary['sad'] = [':(',':[',':-(','=(',':{','://',":\'(",':c']
	emojidictionary['angry'] = ['^^',':@',':S','>:(','-_-']

	for key in emojidictionary.keys():
		dataset.insert(1,key, 0)
	dataset.insert(1,'EMOJINR', 0)

	def features(dictionary, key, dataset):
		cat = []
		for line in dataset['STATUS']:
			words = line.split(" ")
			wordlist = []
			for word in words:
				wordlist.append(word)
				i = 0
				for value in emojidictionary[key]:
					if value in wordlist:
						i += 1
			cat.append(i)
		dataset[key] = cat
		return (dataset[key])
	features(emojidictionary, 'happy', dataset)	
	features(emojidictionary, 'random', dataset)
	features(emojidictionary, 'winking', dataset)
	features(emojidictionary, 'neutral', dataset)			
	features(emojidictionary, 'sad', dataset)
	features(emojidictionary, 'angry', dataset)	
	dataset['EMOJINR'] = (dataset['happy']+dataset['sad']+dataset['angry']+dataset['random']+dataset['neutral']+dataset['winking'])
	del dataset['STATUS']
	return (dataset)
