# Fake News Detector
## Master Thesis in Data Science - KSchool

This is the repository for my Master Thesis in Data Science at KSchool: **Fake News Detector**, a tool powered with Machine Learning to detect, classify and predict fake news from real news in spanish language. The result of this project is a web app as a live demo. **¡Try it!**

## Table of contents

[1. Introduction](#1.-Introduction)

[2. Requirements](#2.-Requirements)

[3. Materials and methodology](#3.-Materials-and-methodology)

[4. Datasets and corporas](#4.-Datasets-and-corporas)

[5. Data transformation](#5.-Data-transformation)

[6. Feature extraction](#6.-Feature-extraction)

[7. Data Exploration Analysis](#7.-Data-Exploration-Analysis)

[8. Models and classifier](#8.-Models-and-classifier)

[9. App](#9.-App)

[10. Conclusions](#10.-Conclusions)

[11. Future work](#11.-Future-work)

[12. References](#12.-References)



## 1. Introduction

Initially I wanted to find a project and theme where I could apply the knowledge just acquired during the master, being able to experience all phases of a Data Science project: **data adquisition**, **exploratory data analysis**, **data and feature engineering**, **multiple modeling** and finally create a **data product**. During the recent crisis due to the covid-19 pandemic, which we are still suffering from, the use of Online Social Media (OSM) increased a lot the spreading of news, reducing barries and able to reach out to more people. But unfortunately social media is **very limited to check the credibility of news**, so the proliferation of fake news increased during this period. Nowadays news articles can be written and spread by anyone with access to Internet, so the spread of fake news and the massive spread of digital misinformation has been identified as a **major threat to democracies**.

There is empirical evidence that false news are spreading significantly "faster, deeper, and more broadly" than the true ones. An **[MIT study](#https://science.sciencemag.org/content/359/6380/1146)** found that the top 1% of **false news cascades diffused to 1,000 - 100,000 people**, whereas the true ones rarely reached more than 1,000 people. This global risk is tangible when we experience it directly with its **influence in elections, threaten democracies**. While such claims are hard to prove, real harm of disinformation has been demonstrated in health and finance. Public opinion can be influenced thanks to the low cost of producing fraudulent websites and high volumes of **software-controlled profiles**, known as social bots. Humans are vulnerable to this manipulations. But... **What about a machine?** under this premise I started this project.

My first approach was to study and research through realted work on this field and subject. There are several kernels developed to classify and distingish between fake and true news, even adding more categories like satire, hate news, conspirancy, etc. Also, most of these kernels applies different Machine Learning models, Neural Netowrks models with proficient results, applying **Natural Language Processing techniques**, Transfer Learning, Ensemble Learning and through diferents datasets and corpus like the **[Fake News Corpus (FNC) from several27](#https://github.com/several27/FakeNewsCorpus)** or many others. But called my attention that all of this work is majority done in English language, so there's territory to conquer in Spanish language!

There are few advances in the fake news detection field in the Spanish language, and this presents not only an opportunity, but also a **problem in the data**. Most of the related work was done by **Juan Pablo Posadas-Durán and his team**, they created their own **[Spanish Fake News Corpus](#https://github.com/jpposadas/FakeNewsCorpusSpanish)**, and this corpus was used for the **Fake News Detection task in the [MEX-A3T Competition](#https://sites.google.com/view/mex-a3t/)** at the **IberLEF 2020 congress**. After their job, there are no big progress on the fake news detection field in Spanish, even most of the fact checkers in Spain as **[Maldito Bulto](#https://maldita.es/malditobulo/)**, **[Newtral](#https://www.newtral.es/)** or **[EFE Verifica](#https://www.efe.com/efe/espana/efeverifica/50001435)** continue their verification work with traditional journalistic techniques in a very effective but in many cases not very efficient way. 

**¿Can we make an efficient too but also effective to detect between Fake and Real news?**



## 2. Requirements

On this project we are using the work environment proposed on this Master, so we'll keep using the Ubuntu Distribution for windows: **Windows Subsystem Linux (WSL)** And more specifically we will use Anaconda distribution with Python 3.6.9. 

### Python Packges required
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- wordcloud
- nltk
- spacy
- scikit-learn
- xgboost
- regex
- [lexical-diversity](#https://pypi.org/project/lexical-diversity/)
- [newspaper3k](#https://pypi.org/project/newspaper3k/)
- [tldextract](#https://pypi.org/project/tldextract/)
- [syltippy](#https://pypi.org/project/syltippy/)
- streamlit

There are some rare packages like **lexical-diversity** to extract relevant lexical features, **newspaper3k** which permits us to extract the text and headline from a newspaper given the url or **syltippy** which permits us to extract syllables from text in spanish. Also we are using **Spacy** pre trained models in Spanish language, we can work locally with the **`es_core_news_lg` large model**, but later on we will need to switch to the **`es_core_news_md` medium model**, because it fits the heroku server size.

Versions of each specific package:

```
pandas==1.1.1

numpy==1.19.1

streamlit==0.65.2

nltk==3.5

regex==2020.7.14

lexical-diversity==0.1.1

newspaper3k==0.2.8

tldextract==2.2.3

syltippy==1.0

scikit-learn==0.23.1

xgboost==1.1.1

spacy==2.3.2
```


## 3. Materials and methodology

------------------


## 4. Datasets and corpora

My main initial concerns were to find a dataset or **corpus of fake news in Spanish language**, or explore ways to create one automatically with **web scraping techniques** or by manually checking the fake and real news, which is discarded because it would involve a lot of time in tasks unrelated to Data Science. There is a big deficiency of data, Corpora and datsets about fake news in Spanish language, fortunately I have been able to access to the **[Spanish Fake News Corpus](#https://github.com/jpposadas/FakeNewsCorpusSpanish)** built by Juan Pablo Durán-Posadas and its team.

Also...


## 5. Data transformation

------------------


## 6. Feature extraction

------------------


## 7. Data Exploration Analysis

------------------


## 8. Models and classifier

------------------


## 9. App

------------------


## 10. Conclusions

------------------


## 11. Future work

------------------


## 12. References

[The spread of low-credibility content by social bots.](#https://pubmed.ncbi.nlm.nih.gov/30459415/)
