<p align="center">
  <img src="https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/wordcloud_fakenews.png">
</p>

<p align="center">
  <img src="https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/wordcloud_detector.png">
</p>

## Master Thesis in Data Science - KSchool

This is the repository for my Master Thesis in Data Science at KSchool: **Fake News Detector**, a tool powered with **Machine Learning** and **NLP** to detect, predict and classify between **Fake News** from **Real news** in **Spanish language**. 

This project resulted on the launch of a **web app** as a live demo. **[¡Try it!](https://es-fake-news-detector.herokuapp.com/)**

## Table of contents

[1. Introduction](#1.-Introduction)

[2. Requirements](#2.-Requirements)

[3. Materials and methodology](#3.-Materials-and-methodology)

[4. Datasets and corporas](#4.-Datasets-and-corporas)

[5. Data transformation](#5.-Data-transformation)

[6. Feature extraction](#6.-Feature-extraction)

[7. Data Exploration Analysis](#7.-Data-Exploration-Analysis)

[8. Models and Classifiers](#8.-Models-and-Classifiers)

[9. App](#9.-App)

[10. Conclusions](#10.-Conclusions)

[11. Future work](#11.-Future-work)

[12. References](#12.-References)



## 1. Introduction

Initially I wanted to find a project and theme where I could apply the knowledge I have just acquired during the master and which would allow me to experience all phases of a Data Science project: **data adquisition**, **exploratory data analysis**, **data and feature engineering**, **multiple modeling** and finally create a **data product**. . The recent crisis caused by the COVID-19 pandemic, which we are still suffering from, increased even more the use of Online Social Media (OSM). Indeed, OSM has contributed to increase exponentially  the spreading of news, reducing barries and able to reach out to more people. But unfortunately social media is **very limited to check the credibility of news**, so the proliferation of Fake News increased as well, specifically due to the COVID-19 crisis. Nowadays news articles can be written and spread by anyone with access to Internet, so the spread of fake news and the massive spread of digital misinformation has been identified as a **major threat to democracies**.

There is empirical evidence that Fake News are spreading significantly "faster, deeper, and more broadly" than the true ones. An **[MIT study](https://science.sciencemag.org/content/359/6380/1146)** found that the top 1% of **false news cascades diffused to 1,000 - 100,000 people**, whereas the true ones rarely reached more than 1,000 people. This global risk is tangible when we experience it directly with its **influence in elections, threatening democracies**. While such claims are hard to prove, **real harm of disinformation has been demonstrated in health and finance**. Public opinion can be influenced thanks to the low cost of producing fraudulent websites and high volumes of **software-controlled profiles**, known as "social bots". Humans are vulnerable to this manipulations. But... **What about a machine?** Under this premise I started this project.

<p align="center">
  <img src="https://media.giphy.com/media/OqAeQrGmU7lS6tENnQ/giphy.gif">
</p>

My first approach was to study and research through related work on this field and subject. There are several kernels developed to classify and distinguish between FAKE and Reliable News, even adding more categories like satire, hate news, conspiracy, etc. Also, most of these kernels applies different Machine Learning models, Neural Networks models, with proficient results, applying **Natural Language Processing techniques**, Transfer Learning, Ensemble Learning and through diferents datasets and corpus like the **[Fake News Corpus (FNC) from several27](https://github.com/several27/FakeNewsCorpus)**. Likewise, in the recent past competitions and challenges have been carried out with the aim of classifying fake news as the **[Fake News Challenge (FNC1)](http://www.fakenewschallenge.org/)**. In English language there are proficient kernels, but what called my attention was the fact that all of this work is majority done in English language, so **there's territory to conquer in Spanish language**!

There are few advances in the fake news detection field in the Spanish language, and this presents not only an opportunity, but also a **problem in the data**. Most of the related work was done by **Juan Pablo Posadas-Durán and his team**, they created their own **[Spanish Fake News Corpus](https://github.com/jpposadas/FakeNewsCorpusSpanish)**, and this corpus was used for the **Fake News Detection task in the [MEX-A3T Competition](https://sites.google.com/view/mex-a3t/)** at the **IberLEF 2020 congress**. After their job, there are no big progress on the fake news detection field in Spanish, even most of the certified Fact Checkers in Spain as **[Maldito Bulto](https://maldita.es/malditobulo/)**, **[Newtral](https://www.newtral.es/)** or **[EFE Verifica](https://www.efe.com/efe/espana/efeverifica/50001435)** continue their verification work with traditional journalistic techniques, which results very effective but in many cases its **not a very efficient** way. 

**¿Can we make an efficient tool but also effective to detect between Fake and Real news?**


# 2. Requirements

On this project we are using the work environment proposed throughout this Master, so we'll keep using the **Ubuntu Distribution** for Aindows: **Windows Subsystem Linux (WSL)**, and more specifically we will use Anaconda distribution with **Python 3.6.9**. 

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
- streamlit
- [langdetect](https://pypi.org/project/langdetect/)
- [lexical-diversity](https://pypi.org/project/lexical-diversity/)
- [newspaper3k](https://pypi.org/project/newspaper3k/)
- [tldextract](https://pypi.org/project/tldextract/)
- [syltippy](https://pypi.org/project/syltippy/)

There are some rare packages like **langdetect** which allows us to detect the language of a given text, **lexical-diversity** to extract relevant lexical features, **newspaper3k** which permits us to extract the text and headline from a newspaper given the URL or **syltippy** which permits us to extract syllables from text in spanish. Also we are using **Spacy** pre trained models in Spanish language, we can work locally with the **`es_core_news_lg` large model**, but later on we will need to switch to the **`es_core_news_md` medium model**, because it fits the Heroku Server size.

Versions of each specific package:

```
pandas==1.1.1

numpy==1.19.1

streamlit==0.65.2

nltk==3.5

regex==2020.7.14

langdetect==1.0.8

lexical-diversity==0.1.1

newspaper3k==0.2.8

tldextract==2.2.3

syltippy==1.0

scikit-learn==0.23.1

xgboost==1.1.1

spacy==2.3.2
```


# 3. Materials and methodology

**Pipeline proposed for the project and the resulting web application**

<p align="center">
  <img src="https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/project_pipeline.png">
</p>

We propose the following pipeline for our news classification problem: **1. Data Acquisition, 2. Data Transformation, 3. Feature Extraction, 4. Classification Algorithms**.

### Data Acquisition
The methodology followed in this step consisted of an extensive search of datasets and corpus of fake news in Spanish, contact with different **certified fact checkers** in Spain and several attempts of **web scraping techniques** on fake news already verified. Finally, three sources of data were obtained, which will be explained in the following section **[Datasets and corpora](#4.-Datasets-and-corpora)**.

### Data Transformation
This phase consists of **extracting quality data** from raw text, specifically Fake News and Real News from our data sources and finally create our final corpus for the next steps. On this process we are facing a **big file of 30 Gb** with **9,408,908 articles**, so we are using big data tools like chunksize for filtering and processing. To achieve our goal we will follow the following process:
  1. Filter articles by credibility category
  2. Detect language the language of the articles and filter them. 
  3. Text cleaning and article selection
  4. Create the final corpus

The Data Transformation process can be replicated following the next notebooks: [FNC Big Data file](https://github.com/pipe11/TFM_fake_news_detector/blob/master/data_transformation/01_FNC_fake_news_big_data.ipynb), [Detecting and filtering by language](https://github.com/pipe11/TFM_fake_news_detector/blob/master/data_transformation/02_spanish_fake_news_filter.ipynb), [Extracting Real Spanish News](https://github.com/pipe11/TFM_fake_news_detector/blob/master/data_transformation/03_reliable_spanish_news.ipynb), [Final corpus](https://github.com/pipe11/TFM_fake_news_detector/blob/master/data_transformation/04_Create_corpus_news_extracted_FNC.ipynb). Also you may have installed **langdetect** library, which allows us to detect the language of a given text, in our case Spanish.

**Warning**: The news file is not available in this repository due to its big size. If you want to replicate everything, download it from here: https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0

### Feature Extraction
Once the final Corpus has been created, the next step is to process the text with **NLP techniques**, applying the **Spacy library** to extract valuable information from the raw text. The select features in this project are **language-independent**, for example, they do not consider specific terms from a language, in this case Spanish. To accomplish this objective, we are going to extract features from 2 categories: **Complexity** and **Stylometric (Stylish)**. This process will be explained at the  **[Feature Extraction](#6.-Feature-extraction)** section.

Also we considere the **TF-IDF transformation** to extract text vectors as features.

The Feature Extraction process can be replicated on this notebook: [Feature Extraction](https://github.com/pipe11/TFM_fake_news_detector/blob/master/feature_extraction/06_Final_notebook_feature_extraction_explained.ipynb). 
To replicate this process, you also need to have installed the following specific libraries: **lexical-diversity** to extract relevant lexical features, **syltippy** which permits us to extract syllables from text in Spanish. **Spacy** pre-trained models in Spanish language, we can work locally with the **`es_core_news_lg` large model**.

### Classification Algorithms

On this step we use the proposed features to classify Fake and Real news. We propose the following classification algorithms:

- Logistic regression as benchmark
- K-nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost

Notebook: [Models and Classification Algorithms](https://github.com/pipe11/TFM_fake_news_detector/blob/master/models/11_final_notebook_models_explained.ipynb)

### Predictor and App

The final step is to pack the model and develop an script and test it with actual news articles and launch predictions to classify Fake and Real News. We also make use of the Streamlit library to test an alpha demo of a web app. These steps can be replicated running the following notebook: [Predictos and streamlit](#https://github.com/pipe11/TFM_fake_news_detector/blob/master/predictors/05_Final_notebook_predictor_explained.ipynb)


# 4. Datasets and corpora

My main initial concerns were to find a dataset or **Corpus of Fake News in Spanish language** or explore ways to create one automatically with **web scraping techniques** or by manually checking the Fake and Real News, which was discarded because it would take a lot of time in tasks unrelated to Data Science. There is a big **deficiency of data**, Corpora and Text Data about fake news in Spanish language, fortunately I have been able to access to the **[Spanish Fake News Corpus](https://github.com/jpposadas/FakeNewsCorpusSpanish)** built by **Juan Pablo Durán-Posadas and its team**.

Our first models were developed with this corpus and it contains **971 news** divided into **491 true news** and **480 fake news**. The corpus covers news from 9 different topics: **Science, Sport, Economy, Education, Entertainment, Politics, Health, Security, and Society**.

We observed in our [Classification Algorithms](#8.-Classification-Algorithms) that this corpus is **insufficient** for our purposes to train an efficient and efective clasiffier and detect between Fake and Real news. So we set ourselves **the goal of expanding the Corpus** with more Fake News and Real News.

## Expand the corpus

### Fake News extraction

So after a very long search trying to find text data and corpus in Spanish language, we reached to the **[Fake News Corpus (FNC) from several27](https://github.com/several27/FakeNewsCorpus)**. This dataset is composed of millions of news articles mostly scraped from a curated list of **1001 domains** from **[open sources](http://www.opensources.co/)**. The dataset is still work in progress and for now and the public version includes only **9,408,908 articles** (745 out of 1001 domains) most of them in English language, but there are some that are in **Spanish**. This corpus has several labels for the news: **Fake News, Satire, Extreme Bias, Conspiracy Theory, State News, Junk Science, Hate News, clickbait, Unreliable, Political and Credible**. We are only interested in **Fake News, Satire and Reliable** labels.

Due to the **size of the CSV file** that contains all articles, we couldn't upload it to this repository. On the **[Data Transformation](#5.-Data-Transformation) section** we will address how to proceed to extract the articles in Spanish language and with the desired labels.

After extracting all the news, we realized that the articles with the **Reliable label**, only contains articles **very short** articles and from a single source, so **we rejected them for our corpus**.

### Real News extraction

To **balance the dataset with Real news**, we wanted to do it with articles from the most realiable Newspaper in Spain, so after a long search, we collected the **[WebHouse Dataset](https://webhose.io/free-datasets/spanish-news-articles/)**, **342.000 articles** in Spanish language. These news were **crawled on 2016** and it is a zip with 342.000 of JSON files. 

It were necessary to **filter articles**, classify them according to the proposed topics and choosing wisely the reliable sources, all these operations are covered in the **[Data Transformation](#5.-Data-Transformation) section**

### Final Corpora

After the extraction of **Fake News** and **Real News** we proceed to unify all the three sources of data to create our definitive corpus which contains **3.974 articles** in Spanish language, divided in **2046 Real News** and **1918 Fake News**. We followed the same structure as proposed by **[J.P. Posadas in its Corpus](https://github.com/jpposadas/FakeNewsCorpusSpanish)**, covering 9 different topics: **Science, Sport, Economy, Education, Entertainment, Politics, Health, Security, and Society**. The meaning of the columns is described next:

Id| Category | Topic | Source | Headline | Text |Link |
--|----------|-------|--------|----------|------|-----
Identifier to each instance. | Category of the news (True or Fake). | Topic related to the news. | Name of the source. | Headline of the news. | Raw text of the news. | URL of the source.


# 5. Data transformation

------------------


# 6. Feature extraction

For a good classification performance we need to extract text features to distinguish between Fake and Real News. We tried to capture grammatical, lexical and stylish metrics and ratios from Fake and Real News, from its Headline and its article content. We hace choose wisely these metrics based on **[This Just In: Fake News Packs a Lot in Title](https://arxiv.org/pdf/1703.09398.pdf)** paper written by **Benjamin D. Horne and Sibel Adalı**.

The content of fake and real news articles is substantially different. here is a significant difference in the content of real and fake news articles. **Real News** articles are **significantly longer** than fake news articles and that fake news articles use **fewer technical words**, **smaller words**, **fewer punctuation**, **fewer quotes**, and **more lexical redundancy**.

Also, fake news articles need a **slightly lower education level to read**, use **fewer analytic words**, have **significantly more personal pronouns**, and use **fewer nouns** and **more adverbs**. These differences illustrate a strong divergence in both the complexity and the style of content. Fake news articles seem to be filled with **less substantial information** demonstrated by having a high amount of redundancy, **fewer analytic words**, and **fewer quotes**.

With this analysis in mind we are selecting our features, which are going to be **features are language-independent**, for example, they do not consider specific terms from a language, in this case Spanish. Our objective is to extract features based on high-level structures. To accomplish this objective, we are going to extract features from 2 categories: **Complexity** and **Stylometric (Stylish)**. Also we are considering **readability** and **perspicuity** scores in Spanish, considering **Fernández Huerta's readabilty score** and **Szigriszt Pazos perspicuity score**


### Headline complexity features

Feature | Description | Type |
--------|-------------|------
words_h | Number of words |Integer
avg_words_sentence_h | Average words per sentence | Float
avg_word_size_h | Average word size | Float
avg_syllables_word_h | Average syllables per word | Float
unique_words_h | Ratio of hapaxes or unique words that only appears once in a text | Float
ttr_h | Type token ratio| Float


### Article complexity features

Feature | Description | Type |
--------|-------------|------
words | Number of words |Integer
Sents | Number of sentences | Integer
avg_words_sentence | Average words per sentence | Float
avg_word_size | Average word size | Float
avg_syllables_word | Average syllables per word | Float
unique_words | Ratio of hapaxes or unique words that only appears once in a text | Float
ttr | Type token ratio| Float
huerta_score | Fernández Huerta's redability score (Reading comprehension of the text), spanish adaptation of the Flesch equation | Float
szigriszt_score | Szigriszt Pazos perspicuity score (Legibility and clarity of the text), a modern spanish adaptation of the Flesch equation | Float


### Article stylometric features

Feature | Description | Type |
--------|-------------|------
mltd | Measure of Textual Lexical Diversity, based on McCarthy and Jarvis (2010) | Float
upper_case_ratio | Uppercase letters to all letters ratio |Float
entityratio | Ratio of named Entities to the text size | Float
quotes | Number of quotes | Float
quotes_ratio | Ratio of quotes marks to text size | Float
propn_ratio | Proper Noun tag frequency | Float
noun_ratio | Noun tag frequency | Float
pron_ratio | Pronoun tag frequency | Float
adp_ratio | Adposition tag frequency | Float
det_ratio | Determinant tag frequency | Float
punct_ratio | Punctuation tag frequency | Float
verb_ratio | Verb tag frequency | Float
adv_ratio | Adverb tag frequency | Float
sym_ratio | Symbol tag frequency | Float


### Term Frequency-Inverse Document Frequency features (TFIDF)

Also we most of our models consisted on extract another different type of features with **TFIDF** transformation. Our objective is to convert a the raw text from articles (Tried with headlines too) to a matrix of TF-IDF features:

**TF-IDF** stands for ***term frequency-inverse document frequency***, and the tf-idf weight is a weight often used in information retrieval and text mining. This is a **statistical measure** used to evaluate **how important a word is to a document in our corpus**. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. 

**TF (Term Frequency)**
Usually, the tf-idf weight is composed by two terms: the first computes the normalized **Term Frequency (TF)**, which is the *number of times a word appears in a document, divided by the total number of words* in that document.

**TF(t)** = (Number of times term t appears in a document) / (Total number of terms in the document).

**IDF (Inverse Document Frequency)**
The second term is the **Inverse Document Frequency (IDF)**, computed as the *logarithm of the number of the documents in the corpus divided by the number of documents* where the specific term appears.

**IDF(t)** = log_e(Total number of documents / Number of documents with term t in it).


# 7. Data Exploration Analysis

## Visualizations

### WordCloud
![WordCloud](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/wordcloud.png)

### Number of Fake News and Real News
This corpus which contains **3.974 articles** in Spanish language, divided in **2046 Real News** and **1918 Fake News**.
![articles_count](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/articles_count.png)

### Article topics
The articles in this corpus covers news from 9 different topics: **Science, Sport, Economy, Education, Entertainment, Politics, Health, Security, and Society**.
![articles_topic](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/articles_topic.png)

## Fake and real news topics
![articles_topic_hue](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/articles_topic_hue.png)

## Boxplots with outliers
Through boxplots we can analyze features distributions and check if there are outliers in our data. And check if there are some text with irregularities or text not corresponding to professionally written articles.

With this revision we observed that there are some articles that **don't have punctuations like points**, which are very important for our ratio and sentences features. Also, there are some articles with **lot of punctuation and short sentences**. When I tried to find them on the corpus and checked the text it was the **results of Formula 1 championships**. We are going to **remove these outliers for our training** and also for the rest of our visualizations.

![boxplots](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/boxplots_outliers.png)

## Boxplots without outliers
![boxplots_nooutliers](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/boxplots_nooutliers.png)

## Correlation matrix
Correlation between features:
![boxplots_nooutliers](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/corr_matrix2.png)

## Univariate Distributions Analysis
With the univariate distribution analysis, we are analyzing one variable at a time. Through this visualization we can observe  central tendency (mean, mode and median) and dispersion: range, variance, maximum, minimum, quartiles (including the interquartile range), and standard deviation.
![univariate](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/univariate_plot.png)

## Principal Component Analysis (PCA)
**Principal Component Analysis** is a dimensionality-reduction method that is often used to reduce the dimensionality of large datasets, by transforming a large set of variables into a smaller one that still contains **most of the information** in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to **trade a little accuracy for simplicity**. Because smaller data sets are easier to visualize and make analyzing data much easier for our **purpose to explore Fake and Real News**.

**Fake News labeled as 0 and Real News as 1**

![pca](https://github.com/pipe11/TFM_fake_news_detector/blob/master/imgs/pca.png)





# 8. Classification Algorithms

------------------


# 9. App

------------------


# 10. Conclusions

------------------


# 11. Future work

------------------


# 12. References


Posadas-Durán, J. P., Gómez-Adorno, H., Sidorov, G., & Escobar, J. J. M. (2019). Detection of fake news in a new corpus for the Spanish language. Journal of Intelligent & Fuzzy Systems, 36(5), 4869-4876.

Aragón, M. E., Jarquín, H., Montes-y-Gómez, M., Escalante, H., Villaseñor-Pineda, L., Gómez-Adorno, H., Bel-Neguix, G., & Posadas-Durán, J. (2020). Overview of MEX-A3T at IberLEF 2020: Fake news and Aggressiveness Analysis case study in Mexican Spanish. In Notebook Papers of 2nd SEPLN Workshop on Iberian Languages Evaluation Forum (IberLEF), Preprint.

[The spread of low-credibility content by social bots.](https://pubmed.ncbi.nlm.nih.gov/30459415/) Chengcheng Shao, Giovanni Luca Ciampaglia, Onur Varol, Kai-Cheng Yang, Alessandro Flammini, Filippo Menczer.

[This Just In: Fake News Packs a Lot in Title, Uses Simpler, Repetitive Content in Text Body, More Similar to Satire than Real News Benjamin D. Horne and Sibel Adalı](https://arxiv.org/pdf/1703.09398.pdf)

