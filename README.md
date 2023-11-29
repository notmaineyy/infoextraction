# Extraction of Important terms from Documents

The aim of this project is to extract important details from documents (news articles) in different lanaguages (English, Chinese, Indonesian) quickly and precisely.

Important details to be extracted:<br>
- People <br>
- Locations<br>
- Organisations<br>
- Relationships<br>

This repository contains codes to train the Indonesian NER model and a webapp incorporating information extraction tools developed. An example of a news article to be uploaded to the webapp has also been added to the repo. 

Tools included in the webapp are:
1. Named-Entity Recognition<br>
<img src="https://github.com/notmaineyy/infoextraction/assets/81574037/62903297-0cf2-4687-8665-ffa1929b761a" width="450"><br>
2. Term Frequency-Inverse Document Frequency <br>
<img src="https://github.com/notmaineyy/infoextraction/assets/81574037/de23e5ed-0da7-4442-b032-27b82f62ca48" width="450"><br>

3. Dependency Parser<br>
<img src="https://github.com/notmaineyy/infoextraction/assets/81574037/842034ed-fe85-4751-bc43-6610c42894f8" width="450"><br>

Most of these tools used spaCy pre-trained models. 

## Getting Started
The setup you need to get the webapp running.
```
git clone https://github.com/notmaineyy/infoextraction.git
cd infoextraction
pip install -r requirements.txt
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
```

