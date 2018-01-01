import pymorphy2
import re
import os
import time
from nltk.tokenize import RegexpTokenizer

import vk_api
import facebook
import requests
from bs4 import BeautifulSoup

import pickle
import numpy as np
from sklearn.preprocessing import normalize
from keras.models import load_model

from functools import lru_cache
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
        # Clear text from punctuation etc.
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
        if len(doc) > 2 or sum([len(a) for a in doc]) > 150:
            self.corpora.append(doc)
            self.labels.append(label)


class ParseClass:
    """Class for getting data from sites, facebook, vk"""

    def __init__(self, redis_obj):
        """
        :param redis_obj: Where to push parsed data
        """
        self.redis = redis_obj
        self.pool_data = {}

    def get_from_redis(self, path, num=1000):
        try:
            return [x.decode("utf-8").strip() for x in self.redis.lrange(path, 0, num)]
        except Exception as e:
            print(e)
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

    def get_posts_fb(self, user_fb):
        """
        :param user_fb: str or int whether you are using token or facebook profile id
        if int, there's no access token used
        :return:
        """
        # You'll need an access token here to do anything.  You can get a temporary one
        # here: https://developers.facebook.com/tools/explorer/
        seq = []

        if type(user_fb) == int:
            user = user_fb
        elif type(user_fb) == str:
            access_token = user_fb
            graph = facebook.GraphAPI(access_token)
            profile = graph.get_object('/me')
            user = profile['id']
        else:
            return seq
        path = f"users_fb:{user}"

        if not self.redis.exists(path) and type(user_fb) == str:
            posts = graph.get_connections(user, 'feed')

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
            if seq:
                self.redis.rpush(path, *seq)
        return self.get_from_redis(path)

    @staticmethod
    def get_wall_vk(owner_id, count, access_token):
        vk_session = vk_api.VkApi(token=access_token)
        vk_tools = vk_api.VkTools(vk_session)
        try:
            n_wall = min(1000, count // 25)
            wall = vk_tools.get_all('wall.get', n_wall, {'owner_id': owner_id}, limit=count)['items']
        except Exception as e:
            print(e)
            wall = []
        return [a['text'] for a in wall]

    def process_owner_vk(self, pool, owner_id: int, owner_type='public', n_wall=100):
        """
        Processing RequestsPool wall.get method on that owner_id
        :param pool: vk_api.VkRequestsPool object
        :param owner_id: int
        :param owner_type: str
        :param n_wall: int
        :return: generated path, db_exists indicator
        if db_exists = 1 then we get that values further
        if db_exists = 0 then we need to put values in db
        if db_exists = -1 then we don't do anything
        """
        if owner_type == 'public':
            path = f"publics_vk:{owner_id}"
            owner_id = -owner_id
        else:
            path = f"users_vk:{owner_id}"
        if not self.redis.exists(path):
            if not pool:
                return path, -1
            else:
                self.pool_data[path] = pool.method("wall.get", {"owner_id": owner_id, "count": n_wall})
                return path, 0
        else:
            return path, 1

    def get_publics_and_their_names(self, user_vk: int, num_publics: int):
        """
        Used to get text definition of publics and their ids to get their post further
        :param user_vk: we are using public vk api, so no need for token here
        :param num_publics: slice list of publics and names
        :return:
        """

        if type(user_vk) == int:
            user_id = user_vk
            vk_session = vk_api.VkApi()
            vk = vk_session.get_api()
            groups = vk.users.getSubscriptions(user_id=user_id, extended=1, fields='members_count', count=200)['items']
        else:
            groups = []

        public_ids = [g['id'] for g in groups if 10000 < g.get('members_count', 0) < 3500000]
        names = [g.get('name', "") for g in groups]

        path1 = f"users_vk_publics_ids:{user_id}"
        path2 = f"users_vk_publics_names:{user_id}"

        if not self.redis.exists(path1) and not self.redis.exists(path2):
            self.redis.rpush(path1, *public_ids)
            self.redis.rpush(path2, *names)
        public_ids = [int(a) for a in self.get_from_redis(path1)][:num_publics]
        names = self.get_from_redis(path2)[:num_publics]
        return public_ids, names


class ResultClass:
    def __init__(self, redis_obj):
        self.categories = ['art', 'politics', 'finances', 'strateg_management', 'law', 'elaboration', 'industry',
                           'education', 'charity', 'public_health', 'agriculture', 'government_management', 'smm',
                           'innovations', 'safety', 'military', 'corporative_management', 'social_safety', 'building',
                           'entrepreneurship', 'sport', 'investitions']
        # path to keras model
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/vk_texts_classifier.h5"))
        self.classifier = load_model(path)
        # noinspection PyProtectedMember
        self.classifier._make_predict_function()
        self.graph = tf.get_default_graph()
        # path to vectorizer object
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/vectorizer.p"))
        self.vectorizer = pickle.load(open(path, "rb"))

        self.texts = []
        self.parse_class = ParseClass(redis_obj)

    def parse_vk(self, user_vk, num_publics, n_wall=100):
        """
        Parse vk profile of user, getting wall, getting public names and ids, getting posts from every
        public id and combine altogether.
        :param user_vk: str or int whether you are using token or vk user id
        if int, there's no access token used
        :param num_publics: slice of list in get publics and their names function
        :param n_wall: slice of list in getting number of posts from wall and publics
        :return:
        """

        if type(user_vk) == int:
            user_id = user_vk
            public_ids, names = self.parse_class.get_publics_and_their_names(user_id, num_publics)
            pool = None
            paths = [self.parse_class.process_owner_vk(pool, user_id, owner_type='user')]
            for public_id in public_ids:
                paths.append(self.parse_class.process_owner_vk(pool, public_id, owner_type='public', n_wall=n_wall))
        elif type(user_vk) == str:
            access_token = user_vk
            vk_session = vk_api.VkApi(token=access_token)
            vk = vk_session.get_api()
            user_id = vk.users.get()[0]['id']
            public_ids, names = self.parse_class.get_publics_and_their_names(user_id, num_publics)

            paths = []
            with vk_api.VkRequestsPool(vk_session) as pool:
                paths.append(self.parse_class.process_owner_vk(pool, user_id, owner_type='user'))
                for public_id in public_ids:
                    paths.append(self.parse_class.process_owner_vk(pool, public_id, owner_type='public', n_wall=n_wall))
        else:
            public_ids, names, paths = [], [], []

        self.texts.extend(names)

        for i, (path, db_exists) in enumerate(paths, 0):
            if not db_exists:
                try:
                    wall = [a.get('text', '') for a in self.parse_class.pool_data[path].result.get('items', [])]
                    self.parse_class.redis.rpush(path, *wall)
                except Exception as e:
                    print(e)
                    wall = []
            elif db_exists:
                wall = self.parse_class.get_from_redis(path, n_wall)
            else:
                wall = []
            self.texts.extend(wall)
            try:
                print(f"{i + 1}-th public have been parsed. ({public_ids[i]})")
            except IndexError:
                pass

    def parse_fb(self, user_fb):
        posts = self.parse_class.get_posts_fb(user_fb)
        self.texts.extend(posts)

    def get_result(self, user_vk, user_fb, generator=False):
        if user_vk:
            print(f"VK Parsing {user_vk}")
            t = time.time()
            self.parse_vk(user_vk, 15)
            print(f"VK Parse completed in {time.time() - t} sec.")
        if user_fb:
            print(f"FB Parsing {user_fb}")
            t = time.time()
            self.parse_fb(user_fb)
            print(f"FB Parse completed in {time.time() - t} sec.")

        if self.texts:
            corpora_class = CorporaClass()
            corpora_class.add_to_corpora(self.texts, '')
            print("Added to corpora.")
            transformed = self.vectorizer.transform(corpora_class.corpora[0])
            print("Transformed corpora.")
            with self.graph.as_default():
                predicted = self.classifier.predict(transformed.toarray())
            verdict = normalize(np.sum(predicted, axis=0).reshape(1, -1))[0]
            return list(zip(self.categories, verdict))
        else:
            print("Zero result.")
            return list(zip(self.categories, np.zeros(len(self.categories))))
