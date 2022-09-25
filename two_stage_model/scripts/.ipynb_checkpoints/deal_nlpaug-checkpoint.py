import json
import sys
import spacy
nlp = spacy.load("en_core_web_sm")

def nlp_aug_basline(train):
	track_ids = list(train.keys())
	for id_ in track_ids:
		new_text = ""
		for i,text in enumerate(train[id_]["nl"]):
			doc = nlp(text)

			for chunk in doc.noun_chunks:
				nb = chunk.text
				break
			train[id_]["nl"][i] = nb+'. '+train[id_]["nl"][i]
			new_text += nb+'.'
			if i<2:
				new_text+=' '
		train[id_]["nl"].append(new_text)
	with open(sys.argv[1].split('.')[-2]+"_nlpaug.json", "w") as f:
		json.dump(train, f,indent=4)


def nlp_aug_modified(train):
	track_ids = list(train.keys())
	desc_dict = {}
	for id_ in track_ids:
		print("Processing id: "+id_)
		train[id_]["subjects"] = []
		temp = set()
		for text in train[id_]["nl"]:
			doc = nlp(text)
			for chunk in doc.noun_chunks:
				nb = chunk.text
				break
			temp.add(nb.strip().lower())

		# for text in train[id_]["nl_other_views"]:
		# 	doc = nlp(text)
		# 	for chunk in doc.noun_chunks:
		# 		nb = chunk.text
		# 		break
		# 	temp.add(nb.strip().lower())
		
		for s in temp:
			train[id_]["subjects"].append(s)
			if(s not in desc_dict.keys()):
				desc_dict[s] = set()
			desc_dict[s].add(id_)

	for id_ in track_ids:
		temp = set()
		for sub in train[id_]["subjects"]:
			temp.update(desc_dict[sub])
		
		train[id_]["targets"] = []
		for uid in temp:
			train[id_]["targets"].append(uid)

			
	with open(sys.argv[1].split('.')[-2]+"_nlpaug_modified.json", "w") as f:
		json.dump(train, f,indent=4)



def main():
    
	with open(sys.argv[1]) as f:
		train = json.load(f)
	
	if(sys.argv[2] == "baseline"):
		print("Applying baseline NLP augmentation")
		nlp_aug_basline(train)
	else:
		print("Applying modified NLP augmentation")
		nlp_aug_modified(train)

if __name__ == "__main__":
    main()