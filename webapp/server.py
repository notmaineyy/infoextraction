import base64
from PIL import Image
from io import BytesIO
import re
from flask import Flask,url_for,render_template,request
from flaskext.markdown import Markdown
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import stem
nltk.download('stopwords')
from nltk.corpus import stopwords
import os
import PyPDF2
from PyPDF2 import PdfFileReader
import numpy as np
from sklearn.feature_extraction import text
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from nltk import tokenize
from nltk.tokenize import sent_tokenize
import stopwordsiso
from os import path
from imageio import imread
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud, ImageColorGenerator

app = Flask(__name__)
Markdown(app)

  
def pdf2text(pdf):
    '''Iterate over pages and extract text'''
    text = ''
    for i in range(pdf.getNumPages()):
        page = pdf.getPage(i)
        text = text + page.extractText()
    return text
 
def stem_tokenize(document):
    '''return stemmed words longer than 2 chars and all alpha'''
    tokens = [stem(w) for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens
 
def tokenize(document):
    '''return words longer than 2 chars and all alpha'''
    tokens = [w for w in document.split() if len(w) > 2 and w.isalpha()]
    return tokens
 
def build_corpus_from_dir(corpus, file_name):
    pdf = PdfFileReader(file_name,'rb')
    document = pdf2text(pdf)
    corpus.append(document)
    return corpus

@app.route('/index')
def index():
    return render_template(
        'index.html',
        data=[{'lang': 'Select Language'}, {'lang': 'English'}, {'lang': 'Chinese'},{'lang': 'Indonesian'}])
	

@app.route('/ner')
def ner():
	return render_template(
        'ner.html',
        data=[{'lang': 'Select Language'}, {'lang': 'English'}, {'lang': 'Chinese'},{'lang': 'Indonesian'}])
 
 
@app.route('/dependency')
def dependency():
	return render_template(
        'dependency.html',
        data=[{'lang': 'Select Language'}, {'lang': 'English'}, {'lang': 'Chinese'},{'lang': 'Indonesian'}])
 
@app.route('/tfidf')
def tfidf():
	return render_template(
        'tfidf.html',
        data=[{'lang': 'Select Language'}, {'lang': 'English'}, {'lang': 'Chinese'},{'lang': 'Indonesian'}])
 
@app.route('/')
def about():
	return render_template(
        'about.html')
    
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
      tf=False
      ner=False
      dep=False
      tool = request.form.getlist("method")
      for i in tool:
        if i=='ner':
          ner=True
        if i=='dep':
          dep=True
        if i=="tf":
          tf=True
      f = request.files.getlist('file[]')
      files= []
      all_text=[]
      numOfFiles=0
      for file in f:
        numOfFiles+=1
        corpus = []
        corpus = build_corpus_from_dir(corpus,file)
        files.append(file.filename)
        test_text =''
        for i in corpus:
          test_text=test_text+i
        all_text.append(test_text)
      lang_chosen = request.form.get('language') 
      
      if lang_chosen == "English":
        stop=stopwords.words('english')
        nlp = spacy.load('en_core_web_trf')
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
      elif lang_chosen == "Indonesian":
        stop=stopwords.words('indonesian')
        nlp = spacy.load('nlp_id_checkpoint_2022_09_22_06')
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
      else: 
        nlp = spacy.load('zh_core_web_trf')
        stop=stopwords("zh") 
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
        
      #NER RESULTS
      if ner==True:
        results=[]
        for i in range(numOfFiles):
          doc = nlp(all_text[i])
          html = displacy.render(doc,style="ent")
          html = html.replace("\n\n","\n")
          HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
          result = HTML_WRAPPER.format(html)
          results.append(result)
        
      if dep==True:
        #DEP RESULTS
        depResult=[]
        for i in range(numOfFiles):
          sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', all_text[i])
          res=[]
          for sentence in sentences:
            doc = nlp(sentence)
            options = {"compact": True, "word_spacing":30}
            html = displacy.render(doc, style='dep',options=options)
            html = html.replace("\n\n","\n")
            HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
            res.append(HTML_WRAPPER.format(html))
          depResult.append(res)
        
      #TF RESULTS
      if tf==True:
        tfIdf = tfIdfVectorizer.fit(corpus)
      indices = np.argsort(tfIdfVectorizer.idf_)[::-1]
      a = tfIdfVectorizer.get_feature_names()
      tfresults=[]
      for i in range(numOfFiles):
          tdm = tfIdfVectorizer.transform([all_text[i]])
          dense = tdm.todense()
          episode = dense[0].tolist()[0]
          phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
          sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
          phrase_score = [(a[word_id], score) for (word_id, score) in sorted_phrase_scores]
          phrase_score_dict = dict(phrase_score)
          wc = WordCloud(mode='RGBA',background_color='white').fit_words(phrase_score_dict)
          file_name = 'wordcloud'+str(i)+'.png'
          save_name = 'static\wordcloud' + str(i) + '.png'
          tfresults.append(file_name)
          wc.to_file(save_name)
          #filename = Image.open(file_name)
          #results.append(filename)
          #filename.show()
      #return render_template('tfresult.html', results=results, files=files, len=numOfFiles)
              
      if ner == True and dep == True and tf==False:
        return render_template('nerdepresult.html', rawtext=all_text, results=results, len=len(all_text), files=files, depResult=depResult)
      
      elif ner == True and tf == True and dep == False:
        return render_template('nertfresult.html', rawtext=all_text, results=results, len=len(all_text), files=files, tfresults=tfresults)
      
      elif tf == True and dep == True and ner==False:
        return render_template('tfdepresult.html', rawtext=all_text, tfresults=tfresults, len=len(all_text), files=files, depResult=depResult)
        
@app.route('/tfuploader', methods = ['GET', 'POST'])
def upload_file_tf():
   if request.method == 'POST':
      f = request.files.getlist('file[]')
      count = request.form["words"]
      if count == '':
        count=50
      count=int(count)
      all_text = []
      files=[]
      numOfFiles=0
      for file in f:
        numOfFiles+=1
        corpus = []
        corpus = build_corpus_from_dir(corpus,file)
        files.append(file.filename)
        test_text =''
        for i in corpus:
          test_text=test_text+i
        all_text.append(test_text)
      lang_chosen = request.form.get('language') 
      if lang_chosen == "English":
        stop=nltk.corpus.stopwords.words('english')
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
      elif lang_chosen == "Indonesian":
        stop=nltk.corpus.stopwords.words('indonesian')
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
      elif lang_chosen == "Chinese":
        stop=stopwordsiso.stopwords('zh') 
        tfIdfVectorizer=TfidfVectorizer(tokenizer=tokenize,stop_words=stop)
      tfIdf = tfIdfVectorizer.fit(corpus)
      indices = np.argsort(tfIdfVectorizer.idf_)[::-1]
      a = tfIdfVectorizer.get_feature_names()
      results=[]
      for i in range(numOfFiles):
          tdm = tfIdfVectorizer.transform([all_text[i]])
          dense = tdm.todense()
          episode = dense[0].tolist()[0]
          phrase_scores = [pair for pair in zip(range(0, len(episode)), episode) if pair[1] > 0]
          sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
          phrase_score = [(a[word_id], score) for (word_id, score) in sorted_phrase_scores]
          phrase_score_dict = dict(phrase_score)
          wc = WordCloud(max_words=count, font_path = 'SourceHanSerif\SourceHanSerifK-Light.otf', mode='RGBA',background_color='white').fit_words(phrase_score_dict)
          file_name = 'wordcloud'+str(i)+'.png'
          save_name = 'static\wordcloud' + str(i) + '.png'
          results.append(file_name)
          wc.to_file(save_name)
          #filename = Image.open(file_name)
          #results.append(filename)
          #filename.show()
      return render_template('tfresult.html', results=results, files=files, len=numOfFiles)
   
@app.route('/neruploader', methods = ['GET', 'POST'])
def upload_file_ner():
   if request.method == 'POST':
      f = request.files.getlist('file[]')
      all_text = []
      files=[]
      numOfFiles=0
      for file in f:
        numOfFiles+=1
        corpus = []
        corpus = build_corpus_from_dir(corpus,file)
        files.append(file.filename)
        test_text =''
        for i in corpus:
          test_text=test_text+i
        all_text.append(test_text)
      lang_chosen = request.form.get('language') 
      if lang_chosen == "English":
        nlp = spacy.load('en_core_web_trf')
      elif lang_chosen == "Indonesian":
        nlp = spacy.load('nlp_id_checkpoint_2022_09_22_06')
      else: 
        nlp = spacy.load('zh_core_web_trf')
      results=[]
      for i in range(numOfFiles):
        doc = nlp(all_text[i])
        html = displacy.render(doc,style="ent")
        html = html.replace("\n\n","\n")
        HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
        result = HTML_WRAPPER.format(html)
        results.append(result)
      return render_template('nerresult.html', rawtext=all_text, results=results, len=len(all_text), files=files)
    
@app.route('/depuploader', methods = ['GET', 'POST'])
def upload_file_dep():
   if request.method == 'POST':
      f = request.files.getlist('file[]')
      files= []
      all_text=[]
      numOfFiles=0
      for file in f:
        numOfFiles+=1
        corpus = []
        corpus = build_corpus_from_dir(corpus,file)
        files.append(file.filename)
        test_text =''
        for i in corpus:
          test_text=test_text+i
        all_text.append(test_text)
      lang_chosen = request.form.get('language') 
      
      if lang_chosen == "English":
        nlp = spacy.load('en_core_web_trf')
      elif lang_chosen == "Indonesian":
        nlp = spacy.load('nlp_id_checkpoint_2022_09_22_06')
      else: 
        nlp = spacy.load('zh_core_web_trf')
      result=[]
      for i in range(numOfFiles):
        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', all_text[i])
        res=[]
        for sentence in sentences:
          doc = nlp(sentence)
          options = {"compact": True, "word_spacing":30}
          html = displacy.render(doc, style='dep',options=options)
          html = html.replace("\n\n","\n")
          HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
          res.append(HTML_WRAPPER.format(html))
        result.append(res)
      return render_template('depresult.html', result=result, len=numOfFiles, files=files)
   
if __name__ == "__main__":
  app.run()