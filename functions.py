import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def calculate_sim(preprocessed_paragraph):
    import torch
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased')

    sentence_embeddings = []
    for sentence in preprocessed_paragraph:
        input_ids = torch.tensor(tokenizer.encode(
            sentence, add_special_tokens=True)).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden_states = outputs[0]

        sentence_embedding = torch.mean(
            last_hidden_states, dim=1).squeeze().tolist()
        sentence_embeddings.append(sentence_embedding)

    similarities = torch.zeros(
        len(preprocessed_paragraph), len(preprocessed_paragraph))

    for i in range(len(preprocessed_paragraph)):
        for j in range(i + 1, len(preprocessed_paragraph)):
            similarity = torch.nn.functional.cosine_similarity(torch.tensor(sentence_embeddings[i]),
                                                               torch.tensor(sentence_embeddings[j]), dim=0)
            similarities[i][j] = similarity

    return similarities


def pre_process(text):
    text = text.lower()
    stemmer = nltk.SnowballStemmer(language='english')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuation = set(string.punctuation)

    sentences = nltk.sent_tokenize(text)
    print("sentences:", len(sentences))
    preprocessed_sentences = []

    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token.lower(
        ) not in stop_words and token not in punctuation and not token.endswith("'s")]

        preprocessed_sentence = ' '.join(stemmed_tokens)
        preprocessed_sentences.append(preprocessed_sentence)

    return preprocessed_sentences


def calc_score(sentences, threshold_counter, text_list, baslik, paragraf):
    stemmer = nltk.SnowballStemmer(language='english')
    stop_words = set(nltk.corpus.stopwords.words("english"))

    idf_paragraf_word = paragraf.lower().split()
    idf_paragraf_word = [word for word in idf_paragraf_word if word.lower() not in stop_words]
    idf_paragraf = ' '.join(idf_paragraf_word)

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    tfidf_separate = tfidf_vectorizer.fit_transform([idf_paragraf])

    total_word_count = len(idf_paragraf_word)

    theme_word_count = int(total_word_count * 0.1)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_separate.toarray().sum(axis=0)

    vocab_df = pd.DataFrame(zip(feature_names, tfidf_values), columns=["vocab", "tfidf_value"])
    vocab_df = vocab_df.sort_values(by="tfidf_value", ascending=False)

    theme_word = list(vocab_df["vocab"].head(theme_word_count))

    theme_word = [stemmer.stem(word) for word in theme_word]

    print("------------------------------:0", theme_word)

    title_stem = [stemmer.stem(word) for word in baslik.lower().split(" ")]

    scores = []

    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        sentence_words = nltk.word_tokenize(sentence_lower)
        sentence_length = len(sentence_words)
        text_list_tokenized = nltk.word_tokenize(text_list[i])
        tagged_words = nltk.pos_tag(text_list_tokenized)

        # P1
        proper_noun_count = 0
        for word, tag in tagged_words:
            if tag == 'NNP':
                proper_noun_count += 1
        p1 = proper_noun_count / sentence_length

        # P2
        numeric_data_count = sum(1 for word in text_list_tokenized if any(
            char.isdigit() for char in word))
        p2 = numeric_data_count / sentence_length

        # P3
        p3 = threshold_counter[i] / (len(sentences) - 1)

        # P4
        headline_word_count = sum(
            1 for word in sentence_words if word in title_stem)
        p4 = headline_word_count / sentence_length

        # P5
        p5_theme_word = sum(1 for word in sentence_words if word in theme_word)

        p5 = p5_theme_word / sentence_length

        score = p1 + p2 + p3 + p4 + p5
        print("Index:", i, "p1:", p1, "p2:", p2, "p3:",
              p3, "p4:", p4, "p5:", p5, "score:", score)
        scores.append(float(score))

    return scores
