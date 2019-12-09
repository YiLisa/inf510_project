import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob


# get billboard hot 100 songs data
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

    # construct the dataframe
    df = pd.DataFrame()
    df['year'] = years
    df['song'] = songs
    df['artists'] = artists
    df['rank'] = ranks

    # separate artists in individual columns, keep the first 4 artists for each song
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


# get lyrics using artist and song name
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


# get artist type: group/person
# if type is person, get gender
def get_artist_type(artist):
    rr = requests.get('http://musicbrainz.org/ws/2/artist/?query=name:{0}&fmt=json'.format(artist))
    try:
        artist_type = json.loads(rr.text)['artists'][0]['type']
    except KeyError:
        return 'Not Found'
    if artist_type == 'Person':
        try:
            gender = json.loads(rr.text)['artists'][0]['gender']
        except KeyError:
            gender = 'Person'
        return gender

    return artist_type


# get artist type for each song
# if there is more than 2 artists, type = collab means it is a collaboration work
def get_song_artists_type(s):
    if s['artist2'] is None:
        return get_artist_type(s['artist1'])
    return 'Collab'


# get genre for each song using song name and artists
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
        except IndexError:
            continue
    return 'Not Found'


# get polarity sentiment score for each lyrics
# score > 0 means lyric is positive, score < 0 means that is negative
def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
