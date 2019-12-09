import argparse
import pandas as pd
import warnings
import get_data as dg
import data_visualization as dv


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-source", choices=["local", "remote", "test"], nargs=1, help="where data should be gotten from")
    args = parser.parse_args()
    location = args.source

    data = 1
    # run local mode
    if location[0] == "local":
        try:
            hot_songs = pd.read_csv('../data/hot_songs.csv')
            print('Accessing full dataset...')
        except IOError:
            try:
                hot_songs = pd.read_csv('../data/hot_songs_sample.csv')
                print('Accessing sample dataset...')
            except IOError:
                data = 0
                print('Data not found, please run in remote or test mode first')

    else:
        # run test mode
        if location[0] == "test":
            # get data by scraping and api requests
            print('Getting representative sample data from billboard.com...')
            hot_songs = dg.get_billboard_data(True, 2009, 2019)
        # run remote mode
        else:
            # get data by scraping and api requests
            print('Getting data from billboard.com...')
            hot_songs = dg.get_billboard_data(False, 2009, 2019)
        print('Getting lyrics data from lyrics.ovh API...')
        hot_songs['lyrics'] = hot_songs.apply(
            lambda x: dg.get_lyrics(x['artist1'], x['artist2'], x['artist3'], x['artist4'], x['song']), axis=1)
        print('Getting genre data from musixmatch API...')
        hot_songs['genre'] = hot_songs.apply(
            lambda x: dg.get_genres(x['song'], x['artist1'], x['artist2'], x['artist3'], x['artist4']), axis=1)
        print('Getting artist type from musicbrainz API...')
        hot_songs['artist_type'] = hot_songs.apply(lambda x: get_song_artists_type(x), axis=1)
        print('Performing sentiment analysis on lyrics...')
        hot_songs['lyrics_sentiment'] = hot_songs['lyrics'].apply(
            lambda x: dg.get_sentiment_score(x) if x != 'Not Found' else 0)
        hot_songs['sentiment'] = hot_songs.apply(lambda x: int(x['lyrics_sentiment'] > 0), axis=1)
        print('saving data...')
        if location[0] == "test":
            hot_songs.to_csv('../data/hot_songs_sample.csv')
        else:
            hot_songs.to_csv('../data/hot_songs.csv')

    # show data visualization graphs
    if data != 0:
        # visualization data analysis result
        print('Showing data visualization...')
        for i in range(2009, 2019):
            dv.get_wordcloud_year(i, hot_songs)
        dv.get_sentiment_trend(2009, 2019, hot_songs)
        dv.get_genre_sentiment(2009, 2019, hot_songs)
        dv.get_type_sentiment(2009, 2019, hot_songs)


if __name__ == "__main__":
    main()
