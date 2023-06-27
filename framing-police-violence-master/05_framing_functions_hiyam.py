from tqdm import tqdm
from ast import literal_eval
from collections import defaultdict
# import gensim.downloader as api
import pickle as pkl
import pandas as pd
import numpy as np
import neuralcoref
import multiprocessing
import json, re, os, spacy, nltk
# from unidecode import unidecode
import argparse
from political_vocabulary import ENTITIES, NOUNS_ADJ, VERBS
nlp = spacy.load("en_core_web_sm")

neuralcoref.add_to_pipe(nlp)

SUBJECTS = ['nsubj', 'nsubjpass']
OBJECTS = ['dobj', 'iobj', 'obj', 'obl', 'advcl', 'pobj']
MODIFIERS = ['amod', 'nn', 'acl']


def coref_preprocess(txt, max_sent_length=1000, max_doc_length=10000):
    # txt = unidecode(txt.strip())
    return ' '.join([sent for sent in nltk.tokenize.sent_tokenize(txt) if len(sent) <= max_sent_length])[
           :max_doc_length]


def get_name_str_set(name):
    return set([n.lower() for n in name.split(" ")])


def get_race_gender_str_set(race_gender):
    try:
        rg = "_".join(race_gender.split(" ")).lower()
        return set([x.strip().lower() for x in open('resources/racial_and_gender_lexicons_basic/%s.txt' % rg, 'r').readlines()])
    except:
        return set()


def token_is_victim(token, name, race, gender):
    victim_set = get_name_str_set(name).union(get_race_gender_str_set(race)).union(get_race_gender_str_set(gender))
    return token.lower_ in victim_set


OFFICER_REGEX = re.compile(
    r'police|officer|\blaw\b|\benforcement\b|\bcop(?:s)?\b|sheriff|\bpatrol(?:s)?\b|\bforce(?:s)?\b|\btrooper(?:s)?\b|\bmarshal(?:s)?\b|\bcaptain(?:s)?\b|\blieutenant(?:s)?\b|\bsergeant(?:s)?\b|\bPD\b|\bgestapo\b|\bdeput(?:y|ies)\b|\bmount(?:s)?\b|\btraffic\b|\bconstabular(?:y|ies)\b|\bauthorit(?:y|ies)\b|\bpower(?:s)?\b|\buniform(?:s)?\b|\bunit(?:s)?\b|\bdepartment(?:s)?\b|agenc(?:y|ies)\b|\bbadge(?:s)?\b|\bchazzer(?:s)?\b|\bcobbler(?:s)?\b|\bfuzz\b|\bpig\b|\bk-9\b|\bnarc\b|\bSWAT\b|\bFBI\b|\bcoppa\b|\bfive-o\b|\b5-0\b|\b12\b|\btwelve\b')

ISRAEL_REGEX = re.compile(r'Israel|Israeli')


def token_is_officer(span):
    return len(OFFICER_REGEX.findall(str(span).lower())) > 0


human_nouns = set(
    pd.read_csv('resources/textbook_analysis/people_terms.csv', names=['noun', 'race/gender', 'category'])[
        'noun'].values)


def token_is_human(token):
    return ((token.lower_ in human_nouns) and (token.pos_ in ['NOUN', 'PRON', 'PROPN']))


def is_victim(cluster, name, race, gender, check_human=False):
    if check_human:
        if not is_human(cluster):
            return False
    for span in cluster:
        for token in span:
            if token_is_victim(token, name, race, gender):
                return True
    return False


def is_officer(cluster, check_human=False):
    if check_human:
        if not is_human(cluster):
            return False
    for span in cluster:
        if token_is_officer(span):
            return True
    return False


def is_human(cluster):
    for span in cluster:
        for token in span:
            if token.pos_ in ['PROPN', 'PRON']:
                return True
            if token.ent_type_ == "PERSON":
                return True
            if token_is_human(token):
                return True
    return False


def token_is_entity(token):
    entities = {}
    for entity_type in ENTITIES:
        if entity_type == 'ethnicities_races':
            for ent in ENTITIES[entity_type]:
                if token.text in ENTITIES[entity_type][ent]:
                    if ent in entities:
                        entities[ent].append(token)
                    else:
                        entities[ent] = [token]

        else:
            if token.text in ENTITIES[entity_type]:
                if token.text in entities:
                    entities[token.text].append(token)
                else:
                    entities[token.text] = [token]
    return entities


def is_entity(cluster):
    entities = {}
    for span in cluster:
        for token in span:
            entities_new = token_is_entity(token)
            if entities_new != {}:
                if entities == {}:
                    entities = entities_new
                else:
                    for k in entities_new:
                        if k in entities:
                            for z in entities_new:
                                entities[k].append(z)
                        else:
                            entities[k] = entities_new[k]

                print('cluster is entity at span: {}'.format(span))
    return entities

def partition_tokens_entities(doc, verbose=True):

    # tokens for every set of entities
    entity_tokens = {}

    for token in doc:
        # print(token)
        entity_tokens_new = token_is_entity(token)
        if entity_tokens == {}:
            entity_tokens = entity_tokens_new
        else:
            for k in entity_tokens_new:
                if k in entity_tokens:
                    entity_tokens[k].extend(entity_tokens_new[k])
                else:
                    entity_tokens[k] = entity_tokens_new[k]

    for cluster in doc._.coref_clusters:

        potential_entities = is_entity(cluster)
        if potential_entities != {}:
            keys = [k for k in potential_entities]
            entity_tokens[keys[0]].extend(list(set([token for span in cluster for token in span])))
    
    if verbose:
        print('entity tokens:', entity_tokens)
    return entity_tokens

def partition_tokens(doc, victimName, victimGender, victimRace, verbose):
    officer_tokens = set()
    victim_tokens = set()

    for token in doc:
        if token_is_officer(token):
            officer_tokens.add(token)
        if token_is_victim(token, victimName, victimRace, victimGender):
            victim_tokens.add(token)

    for cluster in doc._.coref_clusters:
        if is_officer(cluster):
            officer_tokens.update(set([token for span in cluster for token in span]))
        elif is_victim(cluster, victimName, 'ignore_gender', 'ignore_race', check_human=True):
            victim_tokens.update(set([token for span in cluster for token in span]))

    if verbose:
        print('officer_tokens', officer_tokens)
        print('victim_tokens', victim_tokens)
    return officer_tokens, victim_tokens


def get_verbs_and_objects(subject, target_set=set()):
    def get_pobj(prep):
        for child in prep.children:
            if child.dep_ in OBJECTS:
                return child  # .lower_
        return None

    def populate(verb, vo):
        for child in verb.children:
            if child.dep_ in OBJECTS:
                if child in target_set:
                    vo.append((verb, 'TARGET'))
                else:
                    vo.append((verb, child))
            elif child.dep_ == 'prep':
                pobj = get_pobj(child)
                if pobj: vo.append((verb, pobj))
            if child.dep_ in ['conj', 'xcomp']:
                populate(child, vo)

    vo = []
    populate(subject.head, vo)
    return vo



class FrameExtractor(object):

    def __init__(self):
        legal_set = set([x.strip() for x in open('resources/legal_language/legal.txt', 'r').readlines()[7:]])
        self.legal_regex = r'(\b' + r'\b|\b'.join(list(legal_set)) + r'\b)'

        mental_set = set([x.strip() for x in open('resources/empath/mental_illness.txt', 'r').readlines()])
        self.mental_regex = r'(\b' + r'\b|\b'.join(list(mental_set)) + r'\b)'

        crime_set = set([x.strip() for x in open('resources/empath/crime.txt', 'r').readlines()[6:]])
        self.crime_regex = r'(\b' + r'\b|\b'.join(list(crime_set)) + r'\b)'

        # self.word_embeddings = download_word2vec_embeddings()
        self.moral_concepts = self.get_moral_concepts()
        self.political_concepts = {
            'aggression': ['aggression', 'aggressive'],
            'occupation': ['occupied', 'occupation', 'occupying'],
            # 'occupying': ['occupying'],
            'fabricating': ['fabricating', 'fabricated'],
            'struggle': ['struggle', 'struggles', 'struggling'],
            'fail': ['failed', 'failure'],
            'confront': ['confrontation', 'confront', 'confronted', 'confronting'],
            'threaten': ['threats', 'threat', 'threaten'],
            'criminal': ['criminality', 'cruelty', 'crime', 'crimes', 'criminal'],
            'liberate': ['liberate', 'liberated', 'liberation', 'liberating'],
            'oppress': ['oppressed', 'oppression', 'oppressing']
        }

    def extract_frames(self, df):
        extracted_frames = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            text = coref_preprocess(row['text'])
            name = row['name']
            gender = row['gender']
            age = row['age']
            race = row['race']
            weapons = set([x.lower() for x in literal_eval(row['weapons']) if len(x) > 0])
            try:
                doc = nlp(text)
            except Exception as e:
                print(e)
                extracted_frames[i] = {
                    'found.legal_language': None,
                    'found.mental_illness': None,
                    'found.criminal_record': None,
                    'found.fleeing': None,
                    'found.video': None,
                    'found.age': None,
                    'found.gender': None,
                    'found.unarmed': None,
                    'found.armed': None,
                    'found.race': None,
                    'found.official_report': None,
                    'found.interview': None,
                    'found.attack': None,
                    'found.systemic': None,
                    'found.victim_agentless_passive': None,
                    'found.victim_agentive_passive': None,
                    'found.victim_tokens': [],
                    'found.officer_tokens': [],
                    'victim_agentive_passive_heads': [],
                    'victim_agentive_officer_passive_heads': [],
                    'victim_agentless_passive_heads': [],
                    'num_words': 0,
                    'error': True
                }
                continue

            officer_tokens, victim_tokens = partition_tokens(doc, name, gender, race, verbose=False)
            interviews = self.interview(doc, officer_tokens, victim_tokens)
            agent_passive, agent_passive_heads, agent_officer_passive_heads, agentless_passive, agentless_passive_heads = self.victim_passive_frames(
                doc, victim_tokens, officer_tokens)
            extracted_frames[i] = {
                'found.legal_language': self.mentions_legal(text),
                'found.mental_illness': self.mentions_mental(text),
                'found.criminal_record': self.mentions_criminal(text),
                'found.fleeing': self.mentions_fleeing(text),
                'found.video': self.mentions_video(text),
                'found.age': self.mention_age(text, age),
                'found.gender': self.mention_gender(text, gender),
                'found.unarmed': self.is_unarmed(text),
                'found.armed': self.is_armed(doc, weapons), # *
                'found.race': self.mention_race(officer_tokens, victim_tokens, race),
                'found.official_report': interviews[0],
                'found.interview': interviews[1],
                'found.attack': self.mention_attack(doc, officer_tokens, victim_tokens, list(weapons)), # *
                'found.systemic': self.systemic(doc, officer_tokens, victim_tokens), # *
                'found.victim_agentless_passive': agentless_passive,
                'found.victim_agentive_passive': agent_passive,
                'found.victim_tokens': victim_tokens,
                'found.officer_tokens': officer_tokens,
                'victim_agentive_passive_heads': agent_passive_heads,
                'victim_agentive_officer_passive_heads': agent_officer_passive_heads,
                'victim_agentless_passive_heads': agentless_passive_heads,
                'num_words': len(doc),
                'error': False
            }
            self.extract_moral_frames(extracted_frames[i], doc, officer_tokens, victim_tokens) # *

        return pd.merge(df, pd.DataFrame().from_dict(extracted_frames, orient='index'), left_index=True,
                        right_index=True)
    
    def extract_frames_entities(self, sentences):
        extracted_frames = {}
        for i, text in enumerate(sentences):
            print('Sentence: {}'.format(text))
            doc = nlp(text)

            entity_tokens = partition_tokens_entities(doc=doc, verbose=True)
            # self.mention_verbs(doc=doc, entity_tokens=entity_tokens)

            print('EXTRACTING FRAMES ===================================')
            extracted_frames = self.get_general_frames(doc=doc, entity_tokens=entity_tokens, frames=extracted_frames)
            print('=====================================================')
    
    def mentions_legal(self, text):
        match = re.search(self.legal_regex, text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_mental(self, text):
        match = re.search(self.mental_regex, text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_criminal(self, text):
        match = re.search(self.crime_regex, text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_fleeing(self, text):
        match = re.search(
            r'(\bflee(:?ing)?\b|\bfled\b|\bspe(?:e)?d(?:ing)? (?:off|away|toward|towards)|(took|take(:?n)?) off|desert|(?:get|getting|got|run|running|ran) away|pursu(?:it|ed))',
            text.lower())
        if match:
            return match.span()[0]
        return -1

    def mentions_video(self, text):
        match = re.search(r'(body(?: )?cam|dash(?: )?cam)', text.lower())
        if match:
            return match.span()[0]
        return -1

    def mention_age(self, text, age):
        match = re.search(r'\b%s\b' % age, text.lower())
        if match:
            return match.span()[0]
        return -1

    def mention_gender(self, text, gender):
        total_length = len(text)
        text = text[:int(total_length / 3)]
        match = re.search(r'\b(woman|girl|daughter|mother|sister|female)\b', text.lower())
        if gender == 'Male':
            match = re.search(r'\b(man|boy|son|father|brother|male)\b', text.lower())
        if match:
            return match.span()[0]
        return -1

    def is_unarmed(self, text):
        match = re.search(r'unarm(?:ed|ing|s)?', text.lower())
        if match:
            return match.span()[0]
        return -1

    def is_armed(self, doc, weapons=set()):
        weapons = weapons.difference(set(['vehicle', '']))
        for token in doc:
            stripped = re.sub('[^\w]', '', token.lower_)
            if stripped in weapons:
                return token.idx
            if len(stripped) > 0:
                if re.match(r'^arm(ed|ing|s)?$', stripped) and (token.pos_ != 'NOUN'):
                    return token.idx
        return -1

    def mention_race(self, off, vic, race, verbose=False):
        race_set = get_race_gender_str_set(race)
        for token in vic:
            for child in token.head.children:
                if child.lower_ in race_set:
                    if verbose: print(token.lower_, child.lower_, child.dep_)
                    return child.idx
        return -1

    def interview(self, doc, off, vic, verbose=False):
        say = ['say', 'tell', 'explain', 'report', 'answer', 'claim', 'declare', 'reply', 'state',
               'confirm']
        subjects = ['nsubj', 'nsubjpass']

        official_idx = -1
        commoner_idx = -1

        for token in doc:
            if ((token.head.head.lower_ == 'according') or \
                    ((token.head.lemma_ in say) and (token.dep_ in subjects))):
                if (token in off) or (token.lemma_ in ['investigator', 'authority', 'source', 'official']):
                    if verbose: print('official', token)
                    official_idx = token.head.idx
                elif (token not in vic) and ((token.ent_type_ == 'PERSON') or (token.pos_ == 'PRON') or (
                        str(token) in ['man', 'woman', 'he', 'she'])):
                    if verbose: print('commoner', token)
                    commoner_idx = token.head.idx
        return official_idx, commoner_idx

    def mention_verbs(self, doc, entity_tokens):
        for token in doc:
            if token.dep_ == 'nsubj':
                for ent in entity_tokens:
                    if token in entity_tokens[ent]:
                        for verb, obj in get_verbs_and_objects(token, target_set=set()):
                            print(verb, obj)


    def verb_in_target_verbs(self, verb):
        ''' if the verb exists in the set of verbs of interest (target set)'''
        for verb_type in VERBS:
            if verb.text in VERBS[verb_type]:
                return True
        return False

    def get_target_verb_type(self, verb, dep=None):
        ''' This assumes the verb in the target list of verbs. This gets the target verb type (just a title) '''
        for verb_type in VERBS:
            if verb.text in VERBS[verb_type]:
                if dep is not None:
                    if dep == 'nsubj':
                        return verb_type
                    elif dep in OBJECTS:
                        return 'is.{}'.format(VERBS[verb_type][-1])
                else:
                    return verb_type

    def token_in_noun_adjs(self, token):
        ''' if the token exists in the set of nouns/adjectives of interest (target set) '''
        for nounadj in NOUNS_ADJ:
            if token.text in NOUNS_ADJ[nounadj]:
                return True
        return False

    def get_noun_adj_type(self, token):
        ''' This assumes the token in the target list of nouns/adjs. This gets the target noun/adj type (just a title) '''
        for nounadj in NOUNS_ADJ:
            if token.text in NOUNS_ADJ[nounadj]:
                return nounadj

    def token_in_entity_tokens(self, token, entity_tokens):
        for k in entity_tokens:
            if token.text in [t.text for t in entity_tokens[k]]:
                return True
        return False

    # (DONE) VERB with nsubj = ENTITY
    # (DONE) VERB with dobj = ENTITY
    # VERB with dobj = NOUN and NOUN with amod = ENTITY
    # VERB with dobj = noun and noun with amod = ENTITY
    # (DONE) NOUN with amod = ENTITY
    # (DONE) NOUN with prep and prep with dobj = ENTITY
    # (DONE) NOUN with prep and prep with dobj = noun and noun with amod = ENTITY
    # NOUN and amod=ENTITY and head=VERB

    def get_general_frames(self, doc, entity_tokens, frames):
        '''

        :param doc: the parsed sentence
        :param entity_tokens: dictionary containing the entity tokens found inside doc
        :param frames: dictionary of frames in the set of sentences given as input
        :return:
        '''
        for token in doc:

            # VERB with nsubj = ENTITY, VERB with dobj = ENTITY
            if token.dep_ in ['nsubj', 'dobj']:
                if self.token_in_entity_tokens(token=token, entity_tokens=entity_tokens):
                    for verb, obj in get_verbs_and_objects(token, target_set=set()):
                        if self.verb_in_target_verbs(verb=verb):
                            entities = token_is_entity(token)
                            entity_type = [k for k in entities][0]
                            verb_type = self.get_target_verb_type(verb=verb, dep=token.dep_)
                            keydict = '{}.{}'.format(entity_type, verb_type)
                            print('[VERB with {} = ENTITY] FRAME: {}'.format(token.dep_, keydict))
                            if keydict in frames:
                                frames[keydict] += 1
                            else:
                                frames[keydict] = 1

            if token.pos_ == 'NOUN':
                if self.token_in_noun_adjs(token=token):
                    children = [(c, c.dep_) for c in token.children]
                    for t in children:
                        # NOUN with amod = ENTITY
                        if self.token_in_entity_tokens(token=t[0], entity_tokens=entity_tokens) and t[1] in ['amod', 'compound']:
                            entities = token_is_entity(t[0])
                            entity_type = [k for k in entities][0]
                            nounadj_type = self.get_noun_adj_type(token=token)
                            keydict = '{}.{}'.format(entity_type, nounadj_type)
                            print('[NOUN with {} = ENTITY] FRAME: {}'.format(t[1], keydict))
                            if keydict in frames:
                                frames[keydict] += 1
                            else:
                                frames[keydict] = 1

                            # get the head of the NOUN and see if its a VERB
                            ch = [(c, c.dep_) for c in token.head.children]
                            for c in ch:
                                if c[1] in ['xcomp', 'ccomp'] and self.verb_in_target_verbs(c[0]):
                                    verb_type = self.get_target_verb_type(verb=c[0])
                                    keydict = '{}.{}'.format(entity_type, verb_type)
                                    print('[NOUN with amod = ENTITY and NOUN.head = VERB] FRAME: {}'.format(keydict))
                                    if keydict in frames:
                                        frames[keydict] += 1
                                    else:
                                        frames[keydict] = 1

                        # NOUN with prep and prep with dobj = ENTITY -- unity of the PLO
                        # NOUN with prep and prep with dobj = NOUN and amod=ENTITY -- The suffering of the Lebanese people
                        if t[1] == 'prep': # unity of the PLO:::: of is a prep
                            children_sub = [(c, c.dep_) for c in t[0].children]
                            for tsub in children_sub:
                                if self.token_in_entity_tokens(token=tsub[0], entity_tokens=entity_tokens) and tsub[1] in OBJECTS:
                                    entities = token_is_entity(tsub[0])
                                    entity_type = [k for k in entities][0]
                                    nounadj_type = self.get_noun_adj_type(token=token)
                                    keydict = '{}.{}'.format(entity_type, nounadj_type)
                                    print('[NOUN with prep and prep with dobj = ENTITY] FRAME: {}'.format(keydict))
                                    if keydict in frames:
                                        frames[keydict] += 1
                                    else:
                                        frames[keydict] = 1

                                # The suffering of the Lebanese people
                                elif tsub[1] in OBJECTS and tsub[0].pos_ == 'NOUN':
                                    children_sub = [(c, c.dep_) for c in tsub[0].children]
                                    for tsubsub in children_sub:
                                        if tsubsub[1] in ['amod', 'compound'] and self.token_in_entity_tokens(token=tsubsub[0], entity_tokens=entity_tokens):
                                            entities = token_is_entity(tsubsub[0])
                                            entity_type = [k for k in entities][0]
                                            nounadj_type = self.get_noun_adj_type(token=token)
                                            keydict = '{}.{}'.format(entity_type, nounadj_type)
                                            print('[NOUN with prep and prep with dobj = noun and {}=ENTITY] FRAME: {}'.format(tsubsub[1], keydict))
                                            if keydict in frames:
                                                frames[keydict] += 1
                                            else:
                                                frames[keydict] = 1

                #  VERB with dobj = noun and noun with amod = ENTITY
                children = [(c, c.dep_) for c in token.children]
                for t in children:
                    if self.token_in_entity_tokens(token=t[0], entity_tokens=entity_tokens) and t[1] in ['amod', 'compound']:
                        potential_verb = token.head
                        if self.verb_in_target_verbs(verb=potential_verb):
                            entities = token_is_entity(t[0])
                            entity_type = [k for k in entities][0]
                            verb_type = self.get_target_verb_type(verb=potential_verb, dep=token.dep_)
                            keydict = '{}.{}'.format(entity_type, verb_type)
                            print('[VERB with dobj=noun and noun with {} = ENTITY] FRAME: {}'.format(t[1], keydict))
                            if keydict in frames:
                                frames[keydict] += 1
                            else:
                                frames[keydict] = 1


            if token.pos_ == 'PROPN':
                if self.token_in_entity_tokens(token=token, entity_tokens=entity_tokens):
                    head_token = token.head
                    children = [(c, c.dep_) for c in head_token.children]
                    for t in children:
                        if t[1] in ['amod', 'compound'] and self.token_in_noun_adjs(token=t[0]):
                            entities = token_is_entity(token)
                            entity_type = [k for k in entities][0]
                            nounadj_type = self.get_noun_adj_type(token=t[0])
                            keydict = '{}.{}'.format(entity_type, nounadj_type)
                            print('[PROPN with head=noun and {} = ADJ] FRAME: {}'.format(t[1], keydict))
                            if keydict in frames:
                                frames[keydict] += 1
                            else:
                                frames[keydict] = 1


        # check the cluster of the entity if it contains 'possessiveness'
        for entity in entity_tokens:
            for token in entity_tokens[entity]:
                if token.dep_ == 'poss' and token.head.pos_ == 'NOUN' and self.token_in_noun_adjs(token=token.head):
                    entity_type = entity
                    nounadj_type = self.get_noun_adj_type(token=token.head)
                    keydict = '{}.{}'.format(entity_type, nounadj_type)
                    print('[ENTITY cluster with (poss) + NOUN] FRAME: {}'.format(keydict))
                    if keydict in frames:
                        frames[keydict] += 1
                    else:
                        frames[keydict] = 1
        return frames



    def mention_attack(self, doc, off, vic, weapons, verbose=False):
        attack_verbs = ['shoot', 'fire', 'stab', 'lunge', 'confront', 'attack', 'strike', 'injure', 'harm']
        attack_objects = weapons + ['weapon', 'gun']

        for token in doc:
            if token.dep_ == 'nsubj':
                if token in vic:
                    for verb, obj in get_verbs_and_objects(token, target_set=off): # target set is officer
                        if type(obj) == str:
                            if (obj == 'TARGET') and verb.lemma_ in ['drive', 'accelerate', 'advance']:
                                if verbose: print(verb)
                                return verb.idx
                        else:
                            if verb.lemma_ in attack_verbs: # if the verb is an attack verb
                                if verbose: print(verb)
                                return verb.idx
                            if (obj.lemma_ in attack_objects): # OR if the object is an attack object
                                if verbose: print(verb)
                                return verb.idx
                else:
                    for verb, obj in get_verbs_and_objects(token, target_set=off):
                        if (verb.lemma_ in attack_verbs) and (str(obj) == 'TARGET'): # if the verb is an attack verb and the object is in target
                            if verbose: print(verb)
                            return verb.idx
        return -1

    def systemic(self, doc, off, vic, verbose=False):
        def victim_subject(obj):
            for child in obj.head.children:
                if (child.dep_ == 'nsubj') and (child in vic):
                    return True
            return False

        for token in doc:
            if token.dep_ in ['nsubjpass', 'dobj', 'iobj', 'obj']:
                if (token.head.lemma_ in ['shoot', 'kill', 'murder']) and (token not in vic) and (
                        token not in off) and (token.ent_type_ == "PERSON"):
                    if not victim_subject(token):
                        if verbose: print(token.head, token.dep_, token)
                        return token.head.idx
        match = re.search(
            r'(nation(?:[ -])?wide|wide(?:[ -])?spread|police violence|police shootings|police killings|racism|racial|systemic|reform|no(?:[ -])?knock)',
            str(doc).lower())
        if match:
            if verbose: print(match)
            return match.span()[0]
        return -1

    def victim_passive_frames(self, doc, victim_tokens, officer_tokens):
        victim_agent_passive = np.inf
        victim_agentless_passive = np.inf

        victim_agent_passive_heads = []
        victim_agent_officer_passive_heads = []
        victim_agentless_passive_heads = []
        for token in doc:
            if token in victim_tokens:  # victim is the patient / subject here
                if token.dep_ == 'nsubjpass':  # passive

                    has_agent = np.any(np.array([child.dep_ == 'agent' for child in token.head.children]))

                    if has_agent:
                        victim_agent_passive = min(victim_agent_passive, token.head.idx)
                        victim_agent_passive_heads.append((token.head.lower_, token.head.idx))

                        officer_agent = np.any(np.array(
                            [(child.dep_ == 'agent') and (len(set(child.children).intersection(officer_tokens)) > 0) for
                             child in token.head.children]))
                        if officer_agent:
                            victim_agent_officer_passive_heads.append((token.head.lower_, token.head.idx))
                    else:
                        victim_agentless_passive = min(victim_agentless_passive, token.head.idx)
                        victim_agentless_passive_heads.append((token.head.lower_, token.head.idx))

        return victim_agent_passive, victim_agent_passive_heads, victim_agent_officer_passive_heads, victim_agentless_passive, victim_agentless_passive_heads

    def extract_moral_frames(self, frame_dict, doc, off, vic):
        def get_all_verbs_and_modifiers(token):
            vm = []
            if token.dep_ == 'nsubj':
                vm.append(token.head)
            for child in token.children:
                if child.dep_ in MODIFIERS:
                    vm.append(child)
            return vm

        officer_vm = []  # verbs and modifiers
        for token in off: officer_vm.extend(get_all_verbs_and_modifiers(token))

        victim_vm = []  # verbs and modifiers
        for token in vic: victim_vm.extend(get_all_verbs_and_modifiers(token))

        for who, vm in zip(['officer', 'victim'], [officer_vm, victim_vm]):
            for concept in self.moral_concepts:
                matches = [x for x in vm if x.lower_ in self.moral_concepts[concept]]
                if len(matches):
                    try:
                        text = str(doc)
                        idx_1 = matches[0].idx
                        idx_2 = idx_1 + len(matches[0])
                        print(who, concept, matches, text[max(0, idx_1 - 100):min(idx_2 + 100, len(text))])
                    except:
                        print(who, concept, matches)
                frame_dict['%s.%s' % (who, concept)] = len(matches)
                frame_dict['found.%s.%s' % (who, concept)] = matches


    def get_moral_concepts(self):
        morals = defaultdict(set)
        with open('resources/moral_foundations/Enhanced_Morality_Lexicon_V1.1.txt', 'r') as infile:
            for line in infile.readlines():
                split = line.split('|')
                token = split[0][8:]
                moral_foundation = split[4][9:]
                morals[moral_foundation].add(token)

        return dict(morals)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to input file',
                        default='E:/data/prepared/shootings/all_dated.csv')
    args = parser.parse_args()

    # sentence = 'The unity of the PLO and its legitimate struggle to restore usurped rights and lands'
    # doc = nlp(sentence)
    # print(doc._.coref_clusters)
    # print([(c, c.dep_) for c in doc[1].children])
    # for token in doc:
    #     print(token, '==>', token.dep_, '==>', token.head)

    # sentence = 'Christians helped expel the PLO from Lebanon'
    # doc = nlp(sentence)
    # for token in doc:
    #     print(token, '==>', token.dep_, '==>', token.head)
    # print('==================================================================')
    # sentence = 'It is to weaken Shiites who determine the Christian control of Lebanon'
    # doc = nlp(sentence)
    # for token in doc:
    #     print(token, '==>', token.dep_, '==>', token.head)
    #
    # sentence = 'We expect Palestinian fighters to threaten Israeli borders'
    # sentence = 'Israel seized the opportunity of the congested Mediterranean atmosphere and the use of US imperialism for its fleets and planes against Libya because of the Arab weakness, so it uses warplanes as an attempt to discipline the only place Lebanon'
    # sentence = 'Israel\'s aggressive practices against the people'
    # doc = nlp(sentence)
    # for token in doc:
    #     print(token, '==>', token.dep_, '==>', token.head)

    F = FrameExtractor()

    # partition_tokens_entities


    F.extract_frames_entities(sentences=[
        # 'And he appealed to the supporters of the liberation of the south,'
        #                                  ' the western Bekaa and Rashaya, the Lebanese, Arabs and free'
        #                                  ' people in the world to move and contribute with all means and'
        #                                  ' capabilities that support the steadfastness of our resisting'
        #                                  ' people and separate them from the ferocity of the Nazi front of '
        #                                  'the Israeli occupation and its agents and stop the war of mass massacres, '
        #                                  'destruction and starvation.',

        'We Expect Palestinian  fighters to threaten Israeli borders',
                                         # 'Christian forces affiliated with President Amine Gemayel ensure the return of the Palestinians to Lebanon through Jounieh facility',

        'Christians helped expel the PLO from Lebanon',

        'It is to weaken Shiites who determine the Christian control of Lebanon',

        'The unity of the PLO and its legitimate struggle to restore usurped rights and lands',

        'Israel seized the opportunity of the congested Mediterranean atmosphere and the use of US imperialism for its fleets and planes against Libya because of the Arab weakness, so it uses warplanes as an attempt to discipline the only place Lebanon',

                                         'Putting obstacles in front of the continued growth of the role of the Lebanese national resistance to liberate the land and the people of the south',

        'Israel\'s aggressive practices against the people',
        #
        'Our imperial enemies and the Israelis divided us and took advantage of the sacred realms',

        'Terrified by the devastating Israeli war, and by the practices of Israel against its agents',

        'The suffering of the Lebanese people has been prolonged by the Israeli occupation, by Israel\'s inhumane policies and practices',
        #
        #                                  # 'War against Palestinian presence was started by Amal movement in May, caused heavy human losses, in addition to thousands of detainees, kidnapped, and missing Palestinians, '
        #                                  # 'widespread destruction of buildings, homes, and shops'
    ])





# 1986 nahar South Lebanon Conflict
# (1) Expect Palestinian  fighters to threaten Israeli borders
# (2) Christian forces affiliated with President Amine Gemayel ensure the return of the Palestinians
# to Lebanon through Jounieh facility
# (3) Christians helped expel the PLO from Lebanon
# (4) It is to weaken Shiites who determine the Christian control of Lebanon

# The unity of the PLO and its legitimate struggle to restore usurped rights and lands

# (1) Israel seized the opportunity of the congested Mediterranean atmosphere
# and the use of US imperialism for its fleets and planes against Libya because of the Arab weakness,
# (2) so it uses warplanes as an attempt to discipline the only place Lebanon

# Putting obstacles in front of the continued growth of the role
# of the Lebanese national resistance to liberate the land and the people of the south

# israel's aggressive practices against the people


# 1987
# Our imperial enemies and the Israelis divided us and took advantage of the sacred realms

# (1) Terrified by the devastating Israeli war, and by the practices of Israel against its agents
# (2) The suffering of the Lebanese people has been prolonged by the Israeli occupation,
# by Israel's inhumane policies and practices.

# War against Palestinian presence was started by Amal movement in May,
# caused heavy human losses, in addition to thousands of detainees, kidnapped,
# and missing Palestinians, widespread destruction of buildings, homes, and shops
