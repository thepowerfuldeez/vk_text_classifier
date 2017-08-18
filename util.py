import pymorphy2
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class CorporaClass:
    def __init__(self):
        self.corpora = []
        self.vocab = set()

    tokenizer = RegexpTokenizer('\w+')
    morph = pymorphy2.MorphAnalyzer()
    ru_pattern = re.compile("[а-яА-Я]")

    def full_process(text, tokenizer=tokenizer, morph=morph):
        # Clear text from punctuation etc.'''
        tokens = tokenizer.tokenize(text)

        # Turn tokens into normal form excluding non-nouns or verbs
        processed = []
        for token in tokens:
            morphed = morph.parse(token)[0].normal_form
            nf_tag = str(morph.parse(morphed)[0].tag.POS)
            if (nf_tag in ("NOUN", "ADJF", "INFN", "NUMR") and len(token) < 16):
                if len(morphed) == len(re.findall(ru_pattern, morphed)):
                    processed.append(morphed)

        result = " ".join(processed)
        return result

    def add_to_corpora(file_object):
        try:
            doc = []
            for line in file_object:
                try:
                    processed = full_process(line)
                except Exception as e:
                    print(e)
                    processed = ""
                if len(processed):
                    doc.append(processed)
            stoplist = pd.Series(list(itertools.chain(*(a.split() for a in doc)))).value_counts().index[:5]
            print(f"Excluded: {list(stoplist)}")
            accepted_lines = []
            for line in doc:
                accepted_words = []
                for word in line.split():
                    if word not in stoplist:
                        accepted_words.append(word)
                        self.vocab.add(word)
                accepted_lines.append(" ".join(accepted_words))
            self.corpora.append(accepted_lines)
        except:
            pass
        self.vocab = self.vocab - {""}
