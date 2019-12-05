import json
import os
from os import path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from bs4 import BeautifulSoup
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import warnings


def get_billboard_data(istest, year1, year2):
    songs = []
    artists = []
    years = []
    ranks = []
    for year in range(year1, year2):
        url = f'https://www.billboard.com/charts/year-end/{year}/hot-100-songs'
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        if istest:
            contents = soup.find_all('article', {'class': "ye-chart-item"})[:20]
        else:
            contents = soup.find_all('article', {'class': "ye-chart-item"})
        for content in contents:
            song = content.find('div', {'class': "ye-chart-item__title"}).text.strip().strip('\n')
            artist = content.find('div', {'class': "ye-chart-item__artist"}).text.strip().strip('\n')
            rank = content.find('div', {'class': "ye-chart-item__rank"}).text.strip('\n')
            songs.append(song)
            artists.append(artist)
            years.append(year)
            ranks.append(rank)

    df = pd.DataFrame()
    df['year'] = years
    df['song'] = songs
    df['artists'] = artists
    df['rank'] = ranks

    df['artists'] = df['artists'].apply(lambda d: '/'.join('/'.join('/'.join('/'.join(
        '/'.join('/'.join('/'.join('/'.join(d.split('&')).split('Featuring')).split(',')).split(' x ')).split(
            '+')).split('with')).split('With')).split(' X ')))
    df['artists'] = df['artists'].apply(lambda d: d.strip('/'))
    df['artist1'] = df['artists'].apply(lambda d: d.split('/')[0].strip())
    df['artist2'] = df['artists'].apply(
        lambda d: d.split('/')[1].strip() if len(d.split('/')) > 1 else None)
    df['artist3'] = df['artists'].apply(
        lambda d: d.split('/')[2].strip() if len(d.split('/')) > 2 else None)
    df['artist4'] = df['artists'].apply(
        lambda d: d.split('/')[3].strip() if len(d.split('/')) > 3 else None)

    return df


def get_lyrics(artists1, artists2, artist3, artist4, title):
    for artist in [artists1, artists2, artist3, artist4]:
        if artist is None:
            continue
        url = 'https://api.lyrics.ovh/v1/{0}/{1}'.format(artist, title)
        r = requests.get(url)
        if r.status_code == 200:
            lyric = json.loads(r.text)['lyrics']
            if len(lyric) > 0:
                return lyric
    return 'Not Found'


def get_artist_type(artist):
    rr = requests.get('http://musicbrainz.org/ws/2/artist/?query=name:{0}&fmt=json'.format(artist))
    try:
        artist_type = json.loads(rr.text)['artists'][0]['type']
    except:
        return 'Not Found'
    if artist_type == 'Person':
        try:
            gender = json.loads(rr.text)['artists'][0]['gender']
        except KeyError:
            gender = 'Person'
        return gender

    return artist_type


def get_song_artists_type(s):
    if s['artist2'] is None:
        return get_artist_type(s['artist1'])
    return 'Collab'


def get_genres(track, artist1, artist2, artist3, artist4):
    url = 'http://api.musixmatch.com/ws/1.1/track.search?q_track={0}&q_artist={1}&page_size=1&page=1&s_track_rating=DESC&format=json&apikey=290bbab7e5b315c6ae72c308fc42bdf4'
    for artist in [artist1, artist2, artist3, artist4]:
        if artist is None:
            continue
        url_value = url.format(track, artist)
        r = requests.get(url_value)
        try:
            result = json.loads(r.text)['message']['body']['track_list'][0]['track']['primary_genres']['music_genre_list'][0]['music_genre']['music_genre_name']
            return result
        except KeyError:
            continue
    return 'Not Found'


def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_wordcloud_year(year, df):
    stop_words = set(STOPWORDS)
    stop_words.update(["la", "yeah", "oh", "ooh", "na", "en", "eh", "ah", "ha", "woo", "hey", "got", "wanna"])

    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    text = ' '.join(list(df[df['year'] == year]['lyrics']))
    mask_year = np.array(Image.open(path.join(d, f"{year}.png")))
    wordcloud = WordCloud(background_color="white", max_words=100, min_font_size=4, stopwords=stop_words,mask=mask_year,
                          random_state=10).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


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
    plt.figure()
    plt.plot(X, sentiment_scores, label='average score')
    plt.title(f'Sentiment score from {year1} to {year2}')
    plt.ylim(0, 0.15)
    plt.xlabel('year')
    plt.ylabel('sentiment polarity score')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.bar(X, positive_times, label='positive')
    plt.bar(X, negative_times, label='negative')
    plt.title(f'positive vs negative songs from {year1} to {year2}')
    plt.xlabel('year')
    plt.ylabel('number of songs')
    plt.legend(loc='best')
    plt.show()


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


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-source", choices=["local", "remote", "test"], nargs=1, help="where data should be gotten from")
    args = parser.parse_args()
    location = args.source

    data = 1
    if location[0] == "local":
        try:
            hot_songs = pd.read_csv('hot_songs.csv')
            print('Accessing full dataset...')
        except IOError:
            try:
                hot_songs = pd.read_csv('hot_songs_sample.csv')
                print('Accessing sample dataset...')
            except IOError:
                data = 0
                print('Data not found, please run in remote or test mode first')

    else:
        if location[0] == "test":
            # get data by scraping and api requests
            print('Getting representative sample data from billboard.com...')
            hot_songs = get_billboard_data(True, 2009, 2019)
        else:
            # get data by scraping and api requests
            print('Getting data from billboard.com...')
            hot_songs = get_billboard_data(False, 2009, 2019)
        print('Getting lyrics data from lyrics.ovh API...')
        hot_songs['lyrics'] = hot_songs.apply(
            lambda x: get_lyrics(x['artist1'], x['artist2'], x['artist3'], x['artist4'], x['song']), axis=1)
        print('Getting genre data from musixmatch API...')
        hot_songs['genre'] = hot_songs.apply(
            lambda x: get_genres(x['song'], x['artist1'], x['artist2'], x['artist3'], x['artist4']), axis=1)
        print('Getting artist type from musicbrainz API...')
        hot_songs['artist_type'] = hot_songs.apply(lambda x: get_song_artists_type(x), axis=1)
        print('Performing sentiment analysis on lyrics...')
        hot_songs['lyrics_sentiment'] = hot_songs['lyrics'].apply(
            lambda x: get_sentiment_score(x) if x != 'Not Found' else 0)
        hot_songs['sentiment'] = hot_songs.apply(lambda x: int(x['lyrics_sentiment'] > 0), axis=1)
        print('saving data...')
        if location[0] == "test":
            hot_songs.to_csv('hot_songs_sample.csv')
        else:
            hot_songs.to_csv('hot_songs.csv')

    if data != 0:
        # visualization data analysis result
        print('Showing data visualization...')
        for i in range(2009, 2019):
            get_wordcloud_year(i, hot_songs)
        get_sentiment_trend(2009, 2019, hot_songs)
        get_genre_sentiment(2009, 2019, hot_songs)
        get_type_sentiment(2009, 2019, hot_songs)


if __name__ == "__main__":
    main()
