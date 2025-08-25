# Requirements:
# pip install spacy
# python -m spacy download en_core_web_sm

import spacy

# Charger le modèle spaCy pour l'anglais
nlp = spacy.load("en_core_web_sm")

def get_branch_elements(token):
    mots = [token.text]
    for child in token.children:
        mots.extend(get_branch_elements(child))
    return mots

def creation_dict(texte):
    doc = nlp(texte)
    dic = {}
    for token in doc:
        if token.dep_ == 'ROOT':
            i = 0
            for branche in token.children:
                mots = get_branch_elements(branche)
                dic[i] = ' '.join(mots)
                i += 1
                
    return(dic)

def creation_elements_prompt(text):

    #Le premier mot est la racine, les autres chaines de caractères contiennent les groupements de sens.

    dico = creation_dict(text)
    liste = list(dico.values())
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "ROOT" :
            liste = [token.text] + liste
    return liste

# print(creation_elements_prompt("The Seller shall be responsible for loss of and damage to the Customer\u2019s property except the Customer\u2019s Training Means and to the Seller\u2019s personnel at all times while at the Customer\u2019s facilities except in cases of gross negligence or wilful misconduct of the Customer In this case the Customer shall remain responsible for any damage to its Training Means and its personnel including the personnel involved in the Training flights."))