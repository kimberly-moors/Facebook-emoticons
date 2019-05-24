import csv
import nltk

def dictff(dataset):
	"""This function creates a feature set for LIWC categories. Here the function iterates
	over every instance of a data set and counts the appearences of different categories for
	each column; (for example: Filler, Nonflu, Assent, Death, ...).

	Parameters: 
	dataset: a data set consisting of a column ('STATUS') that exists of textual data."""

	#convert dictionary file into dictionary
	file = open ("D:\Thesis\mypersonality_final\LIWC_English.csv",encoding='latin-1')
	document = csv.DictReader(file, delimiter = ',')
	dict_list = []
	for line in document:
		dict_list.append(line)
	itty = {}
	i = 0
	for item in dict_list:
		key = dict_list[i].values()
		value = dict_list[i].values()
		dictionaryX = {key, value}
		i += 1
		for item in dictionaryX:
			lils = (list(item))
			itty[(lils[0])] = lils[1]
	dictionary = itty
	
	LIWC_data = dataset

	for key in dictionary.keys():
		LIWC_data.insert(1, key, 0)

	def features(dictionary, key, dataset):
		"""This fuction creates the information for each feature. It counts the
		number of times a feature appears foreach instance within the data set and 
		places this information into a column that will be added to the data set.

		parameters:
		dictionary = a dictionary in with the keys represent the features and the values
			represent the words that belong to this feature.
		key = the feature that will be focused on.
		dataset = a dataset consisting of a column ('STATUS') that exists of texutal data.
		"""

		cat = []
		for line in LIWC_data['STATUS']:
			line = line.replace('.', ' ')
			line = line.replace(',', ' ')
			line = line.replace('?', ' ')
			line = line.replace('!', ' ')
			line = line.replace('(', ' ')
			line = line.replace(')', ' ')
			line.lower()
			words = line.split(" ")
			wordlist = []
			i = 0
			for word in words: 
				wordlist.append(word)
			for value in wordlist:
				if value in dictionary[key]:
					i += 1
			cat.append(i)
		LIWC_data[key] = cat
		return (LIWC_data[key])

	for key in dictionary.keys():
		features(dictionary, key, LIWC_data)	

	LIWC_data = LIWC_data.drop(columns = ['STATUS'])
	return(LIWC_data)
