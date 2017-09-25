import itertools
import pymorphy2
import re
import os
import pandas as pd
import time
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

from functools import lru_cache
import tqdm
from config import VK_TOKEN, FB_TOKEN
import tensorflow as tf


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
    @lru_cache(maxsize=100000)
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

    def __init__(self, redis_obj):
        self.redis = redis_obj
        self.pool_data = {}

    def get_from_redis(self, path, num=1000):
        try:
            return [x.decode("utf-8").strip() for x in self.redis.lrange(path, 0, num)]
        except Exception as e:
            print(e.args)
            return []

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

    def get_posts_fb(self, user='BillGates'):

        # You'll need an access token here to do anything.  You can get a temporary one
        # here: https://developers.facebook.com/tools/explorer/
        path = f"users_fb:{user}"
        if not self.redis.exists(path):
            access_token = FB_TOKEN

            graph = facebook.GraphAPI(access_token)
            profile = graph.get_object(user)
            posts = graph.get_connections(profile['id'], 'posts')

            seq = []
            while True:
                try:
                    # Perform some action on each post in the collection we receive from
                    # Facebook.
                    for post in posts['data']:
                        msg = post.get('message', '')
                        if msg:
                            _ = seq.append(msg)
                    # Attempt to make a request to the next page of data, if it exists.
                    posts = requests.get(posts['paging']['next']).json()
                except KeyError:
                    # When there are no more pages (['paging']['next']), break from the
                    # loop and end the script.
                    break
            self.redis.rpush(path, *seq)
        return self.get_from_redis(path)

    def get_posts_fb_temp(self, user='BillGates'):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/fb_dump.json"))
        t = json.load(open(path))

        # You'll need an access token here to do anything.  You can get a temporary one
        # here: https://developers.facebook.com/tools/explorer/
        path = f"users_fb:{user}"
        if not self.redis.exists(path):
            seq = []

            if user not in t:
                return seq
            for post in t[user]:
                if post:
                    seq.append(post)
            self.redis.rpush(path, *seq)
        return self.get_from_redis(path)

    def process_owner_vk(self, pool, owner_id, owner_type='public', n_wall=100):
        if owner_type == 'public':
            path = f"publics_vk:{owner_id}"
            owner_id = -owner_id
        else:
            path = f"users_vk:{owner_id}"
        if not self.redis.exists(path):
            self.pool_data[path] = pool.method("wall.get", {"owner_id": owner_id, "count": n_wall})
            return path, 0
        else:
            return path, 1

    @staticmethod
    def get_publics_and_their_names(user_id, num_publics):
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        vk = vk_session.get_api()
        groups = vk.users.getSubscriptions(user_id=user_id, extended=1, fields='members_count', count=200)['items']
        return [g['id'] for g in groups if 10000 < g.get('members_count', 0) < 3500000][:num_publics], \
               [g['name'] for g in groups]


class ResultClass:
    def __init__(self, redis_obj):
        self.categories = ['art', 'politics', 'finances', 'strateg_management', 'law', 'elaboration', 'industry',
                           'education', 'charity', 'public_health', 'agriculture', 'government_management', 'smm',
                           'innovations', 'safety', 'military', 'corporative_management', 'social_safety', 'building',
                           'entrepreneurship', 'sport', 'investitions']
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/vk_texts_classifier.h5"))
        self.classifier = load_model(path)
        self.classifier._make_predict_function()
        self.graph = tf.get_default_graph()
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/vectorizer.p"))
        self.vectorizer = pickle.load(open(path, "rb"))
        self.texts = []
        self.parse_class = ParseClass(redis_obj)

    def parse_vk(self, user_vk, num_publics, n_wall=100):
        vk_session = vk_api.VkApi(token=VK_TOKEN)
        public_ids, names = self.parse_class.get_publics_and_their_names(user_vk, num_publics)
        self.texts.extend(names)

        paths = []
        with vk_api.VkRequestsPool(vk_session) as pool:
            paths.append(self.parse_class.process_owner_vk(pool, user_vk, owner_type='user'))
            for public_id in public_ids:
                paths.append(self.parse_class.process_owner_vk(pool, public_id, owner_type='public', n_wall=n_wall))

        print(self.parse_class.pool_data.keys())

        for i, (path, db_exists) in enumerate(paths, 0):
            if not db_exists:
                try:
                    wall = [a.get('text', '') for a in self.parse_class.pool_data[path].result.get('items', [])]
                    self.parse_class.redis.rpush(path, *wall)
                except Exception as e:
                    print(e)
                    wall = []
            else:
                wall = self.parse_class.get_from_redis(path, n_wall)
            self.texts.extend(wall)
            try:
                print(f"{i + 1}-th public have been parsed. ({public_ids[i]})")
            except IndexError:
                pass
        return 1

    def parse_fb(self, user_fb):
        # TODO: Facebook parsing
        # texts.extend(parse_class.get_posts_fb(user_fb))
        posts = self.parse_class.get_posts_fb_temp(user_fb)
        self.texts.extend(posts)
        if len(posts):
            return 1
        else:
            return 0

    @staticmethod
    def nn_batch_generator(X_data, batch_size):
        samples_per_epoch = X_data.shape[0]
        number_of_batches = samples_per_epoch / batch_size
        counter = 0
        index = np.arange(np.shape(X_data)[0])
        while 1:
            index_batch = index[batch_size * counter:batch_size * (counter + 1)]
            yield X_data[index_batch, :].toarray()
            counter += 1
            if counter > number_of_batches:
                counter = 0

    def get_result(self, user_vk, user_fb, generator=False):
        if user_vk:
            print(f"VK Parsing {user_vk}")
            t = time.time()
            r = self.parse_vk(user_vk, 15)
            print(f"VK Parse completed in {time.time() - t} sec.")
        if user_fb:
            print(f"FB Parsing {user_fb}")
            r = self.parse_fb(user_fb)
            print(f"FB Parse completed with response {r}.")
            if not r and not user_vk:
                return list(zip(self.categories, np.zeros(len(self.categories))))

        corpora_class = CorporaClass()
        corpora_class.add_to_corpora(self.texts, '')
        print("Added to corpora")
        transformed = self.vectorizer.transform(corpora_class.corpora[0])
        print("Transformed corpora.")
        if generator:
            batch_size = 196
            verdict = normalize(np.sum(
                self.classifier.predict_generator(
                    self.nn_batch_generator(transformed, batch_size),
                    transformed.shape[0] // batch_size
                ), axis=0).reshape(1, -1))[0]
            return list(zip(self.categories, verdict))
        else:
            with self.graph.as_default():
                predicted = self.classifier.predict(transformed.toarray())
            verdict = normalize(np.sum(predicted, axis=0).reshape(1, -1))[0]
            return list(zip(self.categories, verdict))
