import pandas as pd
import numpy as np
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, TreebankWordTokenizer, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction import text
import re
import string


nums = ['097','098','099','100','101','102','103','104','105','106','107','108','109','110','111',
            '112','113','114']

prez = ['reagan','bush','clinton','w_bush','obama']

def descr_table(nums):
    congress_df = []
    for num in nums:
        df = pd.read_csv('./data/raw/hein-daily_final/descr_'+num+'.txt', header=0, delimiter='|')
        congress_df.append(df)
    return pd.concat(congress_df)

def spkr_map_table(nums):
    congress_df = []
    for num in nums:
        df = pd.read_csv('./data/raw/hein-daily_final/'+num+'_SpeakerMap.txt', header=0, delimiter='|')
        congress_df.append(df)
    return pd.concat(congress_df)


def speech_table(nums):
    congress_df = []
    for num in nums:
        df = pd.read_csv('./data/raw/hein-daily/speeches/speeches_'+num+'.txt', header=0, delimiter='|',names=['speech_id','speech'], engine='python', error_bad_lines= False)
        congress_df.append(df)
    return pd.concat(congress_df)

def speech_label(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    speech_length = len(text)
    token_text = TreebankWordTokenizer().tokenize(text)
    fdist = FreqDist(token_text)
    hc_freq = (fdist['healthcare'] / speech_length) +(fdist['health'] / speech_length) +(fdist['care'] / speech_length)+(fdist['medicare'] / speech_length) + (fdist['medicaid'] / speech_length) + (fdist['medicine'] / speech_length)
    edu_freq = (fdist['education'] / speech_length) + (fdist['school'] / speech_length) +(fdist['students'] / speech_length)
    fin_freq = (fdist['bankers'] / speech_length) + (fdist['banking'] / speech_length)+(fdist['banks'] / speech_length)+(fdist['loans'] / speech_length)
    if fin_freq > 0.003 and fin_freq > edu_freq and fin_freq > hc_freq:
            return 'Financial'
    elif hc_freq > 0.004 and hc_freq > fin_freq and hc_freq > edu_freq:
            return 'Healthcare'
    elif edu_freq > 0.004 and edu_freq > fin_freq and edu_freq > hc_freq:
            return 'Education'
    else:
        return 'None'

def full_vader_speech(speech):
    """
    Full speech is passed. Tokenizes by sentence. Vader sentiment scores each sentence and
    appends that to a list of scores. Then each compound score is added to a summed value 
    of compound vader scores. Lastly dividing summed compound value by the length of the 
    scores list, you find average compound sentiment score for each speech.
    Print statements can be uncommented to provide sanity check as it did for me.
    """
    vader = SentimentIntensityAnalyzer()
    tokens = sent_tokenize(speech)
    all_scores = []
    cmpd = 0
    for token in tokens:
        score = vader.polarity_scores(token)
        all_scores.append(score)
    for i in all_scores:
        cmpd+=(i['compound'])
    final_tally = cmpd / len(all_scores)
#     print(cmpd)
#     print(len(all_scores))
    return final_tally

def build_sentiment_tables(kind, dataframe):
    dataframe = pd.read_pickle('./data/pickles/'+kind+'/'+dataframe+'.pkl')
    dataframe['sentiment_cmpd']= dataframe.speech.apply(lambda x: full_vader_speech(x))
    return dataframe


def drop_columns(names, dataframe):
    for name in names:
        dataframe.drop(name,axis=1,inplace=True)

def build_edu_era_tables (prez_list):
    for i in prez_list:
            i_df = pd.read_pickle('./data/pickles/era/'+i+'_df.pkl')
            edu_i_df = i_df.loc[(i_df['labels']=='Education')]
            pickle.dump(edu_i_df, open("./data/pickles/era/edu_"+i+"_df.pkl", "wb"))

def build_hc_era_tables (prez_list):
    for i in prez_list:
            i_df = pd.read_pickle('./data/pickles/era/'+i+'_df.pkl')
            hc_i_df = i_df.loc[(i_df['labels']=='Healthcare')]
            pickle.dump(hc_i_df, open("./data/pickles/era/hc_"+i+"_df.pkl", "wb"))

def build_fin_era_tables (prez_list):
    for i in prez_list:
            i_df = pd.read_pickle('./data/pickles/era/'+i+'_df.pkl')
            fin_i_df = i_df.loc[(i_df['labels']=='Financial')]
            pickle.dump(fin_i_df, open("./data/pickles/era/fin_"+i+"_df.pkl", "wb"))

def display_topics(model, feature_names, no_top_words, topic_names=None):
    """
    Displays the top n terms in each topic
    """
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix + 1)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def clean_senate_speech(text):
    '''Make text lowercase, remove text in square brackets, 
    remove punctuation and remove words containing numbers.
    '''

    import re
    text = re.sub("\w*\d\w*", '', text)
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

class NLPProcessor:
    
    def __init__(self, vectorizer_class, tokenizer_function, cleaning_function,lemmer_function):
        self.vectorizer = vectorizer_class
        self.tokenizer = tokenizer_function
        self.cleaning_function = cleaning_function
        self.lemmer = lemmer_function
    
    def fit(self, corpus_list_to_fit):
        cleaned_corpus = list(map(self.cleaning_function, corpus_list_to_fit))
#         print(cleaned_corpus)
        tokenized_list = list(map(self.tokenizer, cleaned_corpus))
#         print(tokenized_list)
        lemmed_list = [' '.join(list(map(self.lemmer, item))) for item in tokenized_list]
#         print(lemmed_list)
        return self.vectorizer.fit(lemmed_list)
    
    def transform(self, corpus_list_to_clean):
        cleaned_corpus = list(map(self.cleaning_function, corpus_list_to_clean))
        tokenized_list = list(map(self.tokenizer, cleaned_corpus))
        lemmed_list = [' '.join(list(map(self.lemmer, item))) for item in tokenized_list]
        return pd.DataFrame(self.vectorizer.transform(lemmed_list).toarray(), 
                            columns=self.vectorizer.get_feature_names())


# def find_topics (dataframe_name, num_of_topics,num_words_per_topic):
# 	nlp.fit(dataframe_name['speech'])
# 	dataframe_name_dtm = nlp.transform(dataframe_name['speech'])
# 	dataframe_name_cv = nlp.vectorizer
# 	nmf_model = NMF(num_of_topics)
# 	doc_topic = nmf_model.fit_transform(dataframe_name_dtm)
# 	return display_topics(nmf_model, dataframe_name_cv.get_feature_names(), num_words_per_topic)


add_stop_words = ['absent','committee','gentlelady', 'hereabout', 'hereinafter', 'hereto' ,
              'herewith', 'nay','pro','sir', 'thereabout', 'therebeforn', 'therein'
              'theretofore', 'therewithal', 'whereat', 'whereinto', 'whereupon', 'yea',
              'adjourn', 'con', 'gentleman', 'hereafter', 'hereinbefore', 'heretofore','month',
              'none', 'republican', 'speak', 'thereafter', 'thereby', 'thereinafter',
              'thereunder', 'today', 'whereby', 'whereof', 'wherever', 'yes','ask','democrat',
              'gentlemen','hereat','hereinto','hereunder','mr','now','say','speaker',
              'thereagainst','therefor','thereof','thereunto','whereabouts','wherefore',
              'whereon','wherewith','yield','can','etc','gentlewoman','hereby','hereof',
              'hereunto','mrs','part','senator','tell','thereat','therefore','thereon',
              'thereupon','whereafter','wherefrom','whereto','wherewithal','chairman',
              'gentleladies','gentlewomen','herein','hereon','hereupon','nai','per','shall',
              'thank','therebefore','therefrom','thereto','therewith','whereas','wherein',
              'whereunder','will']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)