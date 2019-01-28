def emoji_featurizer (dataset):
	#the emoji's placed into a dictionary.
	emojidictionary={}
	emojidictionary['happy']= [':)',':]','XD',':-)','^_^',':D',':P','xD',':}',':>', '=)']
	emojidictionary['random']= ['<3',':$',':o']
	emojidictionary['winking']= [';)',';-)']
	emojidictionary['neutral']= [':|','-.-']
	emojidictionary['sad']= [':(',':[',':-(','=(',':{','://',":\'(",':c']
	emojidictionary['angry']= ['^^',':@',':S','>:(','-_-']
	happy=[]
	random=[]
	winking=[]
	neutral=[]
	sad=[]
	angry=[]

	for key in emojidictionary.keys():
		dataset.insert(1,key, 0)
	dataset.insert(1,'EMOJINR', 0)
		

	for line in dataset['STATUS']:
		words=line.split(" ")
		yoepi=[]
		for word in words:
			yoepi.append(word)
			h=0
			r=0
			w=0
			n=0
			s=0
			a=0
			for value in emojidictionary['happy']:
				if value in yoepi:
					h+=1
			for value in emojidictionary['random']:
				if value in yoepi:
					r+=1
			for value in emojidictionary['winking']:
				if value in yoepi:
					w+=1
			for value in emojidictionary['neutral']:
				if value in yoepi:
					n+=1
			for value in emojidictionary['sad']:
				if value in yoepi:
					s+=1
			for value in emojidictionary['angry']:
				if value in yoepi:
					a+=1

		happy.append(h)
		random.append(r)
		winking.append(w)
		neutral.append(n)
		sad.append(s)
		angry.append(a)
	dataset['happy'] = happy
	dataset['random'] = random
	dataset['winking'] = winking
	dataset['neutral'] = neutral
	dataset['sad'] = sad
	dataset['angry'] = angry
	dataset['EMOJINR'] = (dataset['happy']+dataset['sad']+dataset['angry']+dataset['random']+dataset['neutral']+dataset['winking'])
	del dataset['STATUS']
	return(dataset)
