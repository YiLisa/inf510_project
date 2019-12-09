from os import path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS


# generate wordcloud graphs for each year's lyrics
def get_wordcloud_year(year, df):
    stop_words = set(STOPWORDS)
    stop_words.update(["la", "yeah", "oh", "ooh", "na", "en", "eh", "ah", "ha", "woo", "hey", "got", "wanna"])
    d = '../data/'
    text = ' '.join(list(df[df['year'] == year]['lyrics']))
    mask_year = np.array(Image.open(path.join(d, f"{year}.png")))
    wordcloud = WordCloud(background_color="white", max_words=100, min_font_size=4, stopwords=stop_words, mask=mask_year,
                          random_state=10).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# generate chats to see sentiment trend over years
def get_sentiment_trend(year1, year2, df):
    sentiment_scores = []
    positive_times = []
    negative_times = []
    X = range(year1, year2)
    for i in range(year1, year2):
        sentiment_score = np.mean(df[df['year'] == i]['lyrics_sentiment'])
        sentiment_time = sum(df[df['year'] == i]['sentiment'])
        sentiment_scores.append(sentiment_score)
        positive_times.append(sentiment_time)
        negative_times.append(len(df[df['year'] == i]) - sentiment_time)

    # line chart for average sentiment score over years
    plt.figure()
    plt.plot(X, sentiment_scores, label='average score')
    plt.title(f'Sentiment score from {year1} to {year2}')
    plt.ylim(0, 0.15)
    plt.xlabel('year')
    plt.ylabel('sentiment polarity score')
    plt.legend(loc='best')
    plt.show()

    # bar chart for number of positive/negative lyrics over years
    plt.figure()
    plt.bar(X, positive_times, label='positive')
    plt.bar(X, negative_times, label='negative')
    plt.title(f'positive vs negative songs from {year1} to {year2}')
    plt.xlabel('year')
    plt.ylabel('number of songs')
    plt.legend(loc='best')
    plt.show()


# generate line chart on sentiment trend over years of top 4 genres
def get_genre_sentiment(year1, year2, df):
    pop = []
    hip = []
    RB = []
    electronic = []
    for y in range(year1, year2):
        pop_score = np.mean(df[df['year'] == y][df['genre'] == 'Pop']['lyrics_sentiment'])
        hip_score = np.mean(df[df['year'] == y][df['genre'] == 'Hip Hop/Rap']['lyrics_sentiment'])
        country_score = np.mean(
            df[df['year'] == y][df['genre'] == 'Contemporary R&B']['lyrics_sentiment'])
        dance_score = np.mean(df[df['year'] == y][df['genre'] == 'Electronic']['lyrics_sentiment'])
        pop.append(pop_score)
        hip.append(hip_score)
        RB.append(country_score)
        electronic.append(dance_score)
    x = range(year1, year2)
    plt.figure()
    plt.plot(x, pop, label='Pop')
    plt.plot(x, hip, label='Hip Hop/Rap', linestyle=':')
    plt.plot(x, RB, label='R&B', linestyle='-.')
    plt.plot(x, electronic, label='Electronic', linestyle='--')
    plt.title(f'Sentiment score from {year1} to {year2} for top 4 genres')
    plt.xlabel('year')
    plt.ylabel('sentiment polarity score')
    plt.legend(loc='best')
    plt.show()


# generate line chart on sentiment trend over years of different artists types
def get_type_sentiment(year1, year2, df):
    female = []
    male = []
    group = []
    collab = []
    for y in range(year1, year2):
        f_score = np.mean(df[df['year'] == y][df['artist_type'] == 'female']['lyrics_sentiment'])
        m_score = np.mean(df[df['year'] == y][df['artist_type'] == 'male']['lyrics_sentiment'])
        g_score = np.mean(df[df['year'] == y][df['artist_type'] == 'Group']['lyrics_sentiment'])
        c_score = np.mean(df[df['year'] == y][df['artist_type'] == 'Collab']['lyrics_sentiment'])
        female.append(f_score)
        male.append(m_score)
        group.append(g_score)
        collab.append(c_score)
    x = range(year1, year2)
    plt.figure()
    plt.plot(x, female, label='Female')
    plt.plot(x, male, label='Male', linestyle=':')
    plt.plot(x, group, label='Group', linestyle='-.')
    plt.plot(x, collab, label='Collab', linestyle='--')
    plt.title(f'Sentiment score from {year1} to {year2} for different artist types')
    plt.xlabel('year')
    plt.ylabel('sentiment polarity score')
    plt.legend(loc='best')
    plt.show()
