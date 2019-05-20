def emoji_featurizer (dataset):
	"""This function creates a featureset of emoticons. Here the function iterates
	over every instance of a dataset and counts the number of emoticons for each column;
	(happy, random, winking, neutral, sad, angry). Additionally, a column is created that 
	represent the sum of all the emoticons.

	Parameters: 
	dataset =  a dataset consisting of a column ('STATUS') that exists of texutal data, possibly 
	containing emoticons.
	"""

	emojidictionary = {}
	emojidictionary['Happy'] = [':)',':]','XD',':-)','^_^',':D',':P','xD',':}',':>', '=)']
	emojidictionary['Remaining'] = ['<3',':$',':o']
	emojidictionary['Winking'] = [';)',';-)']
	emojidictionary['Neutral'] = [':|','-.-']
	emojidictionary['Sad'] = [':(',':[',':-(','=(',':{','://',':\'(',':c']
	emojidictionary['Angry'] = ['^^',':@',':S','>:(','-_-']

	emoji_data = dataset
	for key in emojidictionary.keys():
		emoji_data.insert(1, key, 0)
	emoji_data.insert(1,'EMOJINR', 0)

	def features(dictionary, key, emoji_data):
		"""This fuction creates the information for each feature. It counts the
		number of times a feature appears within each line within the data set and 
		places this information into a column that will be added to the data set.

		parameters:
		dictionary = a dictionary in which the keys represent the features and the values
			represent the words that belong to these features.
		key = the feature that will be focused on.
		dataset = a data set consisting of a column ('STATUS') that exists of textual data, possibly 
			containing emoticons.

		"""
		
		cat = []
		for line in emoji_data['STATUS']:
			line = line.replace(',', ' ')
			line = line.replace('.', ' ')
			words = line.split(" ")
			i = 0
			wordlist = []
			for word in words:
				wordlist.append(word)
			for value in wordlist:
				if value in emojidictionary[key]:
					i += 1
			cat.append(i)
		emoji_data[key] = cat
		return (emoji_data[key])

	for key in emojidictionary.keys():
		features(emojidictionary, key, emoji_data)	

	emoji_data['EMOJINR'] = (emoji_data['Happy'] + emoji_data['Sad'] + emoji_data['Angry'] + emoji_data['Remaining'] + emoji_data['Neutral'] + emoji_data['Winking'])
	emoji_data =  emoji_data.drop(columns = ['STATUS'])
	return (emoji_data)
