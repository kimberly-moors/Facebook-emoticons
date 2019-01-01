def emoji_featurizer (dataset):
	#the emoji's placed into a dictionary.
	emojidictionary={}
	emojidictionary['happy']= [':)',':]','XD',':-)','^_^',':D',':P','xD',':}',':>', '=)']
	emojidictionary['random']= ['<3',':$',':o']
	emojidictionary['winking']= [';)',';-)']
	emojidictionary['neutral']= [':|','-.-']
	emojidictionary['sad']= [':(',':[',':-(','=(',':{','://',":\'(",':c']
	emojidictionary['angry']= ['^^',':@',':S','>:(','-_-']
	emojis=[]
	happy=[]
	random=[]
	winking=[]
	neutral=[]
	sad=[]
	angry=[]

	
	dataset.insert(1,'HAPPY',0)
	dataset.insert(1,'RANDOM',0)
	dataset.insert(1,'WINKING',0)
	dataset.insert(1,'NEUTRAL',0)
	dataset.insert(1,'SAD',0)
	dataset.insert(1,'ANGRY',0)
	dataset.insert(1,'EMOJINR', 0)
		
	for value in emojidictionary.values():
		for v in value:
			emojis.append(v)

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
				else:
					None
			for value in emojidictionary['random']:
				if value in yoepi:
					r+=1
				else:
					None
			for value in emojidictionary['winking']:
				if value in yoepi:
					w+=1
				else:
					None
			for value in emojidictionary['neutral']:
				if value in yoepi:
					n+=1
				else:
					None
			for value in emojidictionary['sad']:
				if value in yoepi:
					s+=1
				else:
					None
			for value in emojidictionary['angry']:
				if value in yoepi:
					a+=1
				else:
					None

		happy.append(h)
		random.append(r)
		winking.append(w)
		neutral.append(n)
		sad.append(s)
		angry.append(a)
	dataset['HAPPY'] = happy
	dataset['RANDOM'] = random
	dataset['WINKING'] = winking
	dataset['NEUTRAL'] = neutral
	dataset['SAD'] = sad
	dataset['ANGRY'] = angry
	dataset['EMOJINR'] = (dataset['HAPPY']+dataset['SAD']+dataset['ANGRY'])
	del dataset['STATUS']
	return(dataset)