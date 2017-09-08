import itertools
import pymorphy2
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer

import vk_api
import facebook
import requests
from bs4 import BeautifulSoup

import pickle
import json
import numpy as np
from sklearn.preprocessing import normalize
from keras.models import load_model

import os
import time
import tqdm
from config import VK_TOKEN, FB_TOKEN


class CorporaClass:
    """Class for setting up corpora"""

    def __init__(self):
        self.corpora = []
        self.vocab = set()
        self.labels = []

    tokenizer = RegexpTokenizer('\w+')
    morph = pymorphy2.MorphAnalyzer()
    ru_pattern = re.compile("[а-яА-Я]")

    @staticmethod
    def full_process(text, tokenizer=tokenizer, morph=morph, ru_pattern=ru_pattern):
        # Clear text from punctuation etc.'''
        tokens = tokenizer.tokenize(text)

        # Turn tokens into normal form excluding non-nouns or verbs
        processed = []
        for token in tokens:
            morphed = morph.parse(token)[0].normal_form
            nf_tag = str(morph.parse(morphed)[0].tag.POS)
            if nf_tag in ("NOUN", "ADJF", "INFN", "NUMR") and len(token) < 16:
                if len(morphed) == len(re.findall(ru_pattern, morphed)):
                    processed.append(morphed)

        return processed

    def add_to_corpora(self, file_object, label):
        doc = []
        for line in file_object:
            try:
                processed = line
            except Exception as e:
                print(e)
                processed = ""
            if len(processed.split()) > 2:
                doc.append(processed)
        if len(doc) > 2:
            self.corpora.append(doc)
            self.labels.append(label)

    def process_corpora(self):
        all_words = []
        for doc in tqdm.tqdm(self.corpora):
            all_words.extend(list(itertools.chain(*(a.split() for a in doc))))
        vc = pd.Series(all_words).value_counts()
        stoplist = vc.index[:20].tolist() + vc.index[vc.values == 1].tolist()
        new_corpora = []
        for doc in self.corpora:
            accepted_lines = []
            for line in doc:
                accepted_words = []
                for word in line.split():
                    if word not in stoplist:
                        accepted_words.append(word)
                        self.vocab.add(word)
                accepted_lines.append(" ".join(accepted_words))
            new_corpora.append(accepted_lines)
        self.corpora = new_corpora
        self.vocab = self.vocab - {""}


class ParseClass:
    """Class for getting data from sites, facebook, vk"""

    def __init__(self):
        pass

    @staticmethod
    def get_all_links(url):
        base_url = "/".join(url.split("/")[:3])

        def check_valid(urls):
            """Only full links containing base url"""
            base_url_name = base_url.split("/")[-1].split(".")[0]
            valid = list(filter(lambda x: base_url_name in x, urls))
            wo_base_url = list(filter(lambda x: base_url_name not in x and "http" not in x, urls))
            ext = []
            for link in wo_base_url:
                if link.startswith("/"):
                    ext.append(base_url + link)
                else:
                    ext.append(f"{base_url}/{link}")
            return valid + ext

        def recursive_url(url):
            """Recursively finds the urls"""
            try:
                page = requests.get(url).text
            except:
                page = requests.get(base_url + url).text
            soup = BeautifulSoup(page, "lxml")
            return set([item.get('href', '') for item in soup.find_all('a')])

        def get_links(url):
            page = requests.get(url).text
            soup = BeautifulSoup(page, "lxml")
            links = set([item.get('href', '') for item in soup.find_all('a')])
            new_links = links
            for link in links.copy():
                try:
                    new_links.update(recursive_url(link))
                except:
                    pass
            return check_valid(new_links)

        return check_valid(get_links(url))

    @staticmethod
    def get_posts_fb(user='BillGates'):

        # You'll need an access token here to do anything.  You can get a temporary one
        # here: https://developers.facebook.com/tools/explorer/
        path = f"assets/corpora_cached_fb_users/{user}.txt"
        if not os.path.exists(path):
            access_token = FB_TOKEN

            graph = facebook.GraphAPI(access_token)
            profile = graph.get_object(user)
            posts = graph.get_connections(profile['id'], 'posts')

            with open(path, 'w') as f:
                while True:
                    try:
                        # Perform some action on each post in the collection we receive from
                        # Facebook.
                        for post in posts['data']:
                            msg = post.get('message', '')
                            if msg:
                                _ = f.write(f"{msg}\n")
                        # Attempt to make a request to the next page of data, if it exists.
                        posts = requests.get(posts['paging']['next']).json()
                    except KeyError:
                        # When there are no more pages (['paging']['next']), break from the
                        # loop and end the script.
                        break
        with open(path) as f:
            return list(f)

    @staticmethod
    def get_posts_fb_temp(user='BillGates'):
        t = json.load(open("assets/fb_dump.json"))

        # You'll need an access token here to do anything.  You can get a temporary one
        # here: https://developers.facebook.com/tools/explorer/
        path = f"assets/corpora_cached_fb_users/{user}.txt"
        if not os.path.exists(path):
            with open(path, 'w') as f:
                for post in t[user]:
                    if post:
                        _ = f.write(f"{post}\n")
        with open(path) as f:
            return list(f)

    @staticmethod
    def getallwall(kwargs, n=None):
        """Get all texts from wall generator"""
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        tools = vk_api.VkTools(vk_session)
        if n:
            try:
                wall_posts = tools.get_all_iter("wall.get", 100, values=kwargs, limit=n)
                for i, post in enumerate(wall_posts, 1):
                    if i > n:
                        break
                    yield post['text']
            except:
                print("Going slow parse")
                wall_posts = tools.get_all_iter("wall.get", 25, values=kwargs, limit=n)
                for i, post in enumerate(wall_posts, 1):
                    if i > n:
                        break
                    yield post['text']
        else:
            try:
                wall_posts = tools.get_all_iter("wall.get", 100, values=kwargs)
                for post in wall_posts:
                    yield post['text']
            except:
                print("Going slow parse")
                wall_posts = tools.get_all_iter("wall.get", 25, values=kwargs)
                for post in wall_posts:
                    yield post['text']

    def process_owner_vk(self, owner_id, owner_type='public', n_wall=None):
        if owner_type == 'public':
            path = f"assets/corpora_cached_vk_publics/{owner_id}.txt"
            owner_id = -owner_id
        else:
            path = f"assets/corpora_cached_vk_users/{owner_id}.txt"
        if not os.path.exists(path):
            wall = self.getallwall({"owner_id": owner_id}, n_wall)
            with open(path, "w") as f:
                for post in wall:
                    if post:
                        _ = f.write(f"{post}\n")
        with open(path) as f:
            return list(f)

    @staticmethod
    def get_publics(user_id, num_publics):
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        vk = vk_session.get_api()
        groups = vk.groups.get(user_id=user_id, extended=1, fields='members_count', count=30)['items']
        return [g['id'] for g in groups if g.get('members_count', 0) > 10000][:num_publics]


class ResultClass:
    def __init__(self):
        self.categories = json.load(open("assets/categories.json"))
        self.classifier = load_model("assets/vk_texts_classifier.h5")
        self.vectorizer = pickle.load(open("assets/vectorizer.p", "rb"))
        self.texts = []
        self.parse_class = ParseClass()
        self.publics_dict = json.load(open("assets/publics_dict.json"))

    def parse_vk(self, user_vk, parse=True):
        self.texts.extend(self.parse_class.process_owner_vk(user_vk, owner_type='user'))
        if parse is True:
            public_ids = self.publics_dict.get(user_vk, self.parse_class.get_publics(user_vk, 5))
            self.publics_dict[user_vk] = public_ids
        else:
            public_ids = self.publics_dict.get(user_vk, [])
        for i, public_id in enumerate(public_ids, 1):
            self.texts.extend(self.parse_class.process_owner_vk(public_id, owner_type='public', n_wall=800))
            print(f"{i}-th public have been parsed.")

    def parse_fb(self, user_fb):
        # texts.extend(parse_class.get_posts_fb(user_fb))
        self.texts.extend(self.parse_class.get_posts_fb_temp(user_fb))

    def get_result(self, user_vk, user_fb):
        if user_vk:
            self.parse_vk(user_vk)
            print("VK Parse completed.")
        if user_fb:
            self.parse_fb(user_fb)
            print("FB Parse completed.")

        corpora_class = CorporaClass()
        corpora_class.add_to_corpora(self.texts, '')
        print("Added to corpora")
        transformed = self.vectorizer.transform(corpora_class.corpora[0]).toarray()
        print("Transformed corpora.")
        verdict = normalize(np.sum(self.classifier.predict(transformed),
                                   axis=0).reshape(1, -1))[0]
        json.dump(self.publics_dict, open("assets/publics_dict.json", "w"))
        return list(zip(self.categories, verdict))
