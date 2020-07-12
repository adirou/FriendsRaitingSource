import os
import re

import pandas as pd
import urllib3
from bs4 import BeautifulSoup
from imdb import IMDb
import collections

pd.set_option('display.max_colwidth', -1)

seasons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

season1_episodes = ['0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0110', '0111', '0112',
                    '0113', '0114', '0115', '0116', '0117', '0118', '0119', '0120', '0121', '0122', '0123', '0124']
season2_episodes = ['0201', '0202', '0203', '0204', '0205', '0206', '0207', '0208', '0209', '0210', '0211', '0212-0213',
                    '0214', '0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222', '0223', '0224']
season3_episodes = ['0301', '0302', '0303', '0304', '0305', '0306', '0307', '0308', '0309', '0310', '0311', '0312',
                    '0313', '0314', '0315', '0316', '0317', '0318', '0319', '0320', '0321', '0322', '0323', '0324',
                    '0325']
season4_episodes = ['0401', '0402', '0403', '0404', '0405', '0406', '0407', '0408', '0409', '0410', '0411', '0412',
                    '0413', '0414', '0415', '0416', '0417', '0418', '0419', '0420', '0421', '0422', '0423']
season5_episodes = ['0501', '0502', '0503', '0504', '0505', '0506', '0507', '0508', '0509', '0510', '0511', '0512',
                    '0513', '0514', '0515', '0516', '0517', '0518', '0519', '0520', '0521', '0522', '0523']
season6_episodes = ['0601', '0602', '0603', '0604', '0605', '0606', '0607', '0608', '0609', '0610', '0611', '0612',
                    '0613', '0614', '0615-0616', '0617', '0618', '0619', '0620', '0621', '0622', '0623', '0624']
season7_episodes = ['0701', '0702', '0703', '0704', '0705', '0706', '0707', '0708', '0709', '0710', '0711', '0712',
                    '0713', '0714', '0715', '0716', '0717', '0718', '0719', '0720', '0721', '0722', '0723', '0724']
season8_episodes = ['0801', '0802', '0803', '0804', '0805', '0806', '0807', '0808', '0809', '0810', '0811', '0812',
                    '0813', '0814', '0815', '0816', '0817', '0818', '0819', '0820', '0821', '0822', '0823']
season9_episodes = ['0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910', '0911', '0912',
                    '0913', '0914', '0915', '0916', '0917', '0918', '0919', '0920', '0921', '0922', '0923-0924']
season10_episodes = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012',
                     '1013', '1014', '1015', '1016', '1017-1018']

girls = ['rachel', 'monica', 'phoebe']
boys = ['ross', 'chandler', 'joey']

most_common_location = [
    "monica and chandler's apartment",
    "central perk",
    "joey's apartment",
    "ross's apartment",
    "the hallway",
    "phoebe's apartment",
    "a restaurant",
    "ross and rachel's apartment",
    "rachel's office",
    "the hospital"]

show_episodes_names = [season1_episodes, season2_episodes, season3_episodes, season4_episodes, season5_episodes,
                       season6_episodes, season7_episodes, season8_episodes, season9_episodes, season10_episodes]


def get_url_text(url):
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    html_page = r.data.decode('utf-8', errors='ignore')
    soup = BeautifulSoup(html_page, features='html.parser')
    return soup.get_text()


def download_all_episodes():
    if not os.path.isdir('data'):
        os.mkdir('data')

    for season in seasons:
        season_dir_path = 'data/season' + str(season)
        if not os.path.isdir(season_dir_path):
            os.mkdir(season_dir_path)
        episodes_names = show_episodes_names[season - 1]
        for ep_name in episodes_names:
            filename = season_dir_path + f'/{ep_name}.txt'
            if not os.path.isfile(filename):
                url = f'https://fangj.github.io/friends/season/{ep_name}.html'
                url_text = get_url_text(url)
                with open(filename, "w") as f:
                    f.write(url_text)


def extract_scene_text(line):
    is_scene = False
    scene_text = None

    if line.startswith('[Scene:') or line.startswith('(Scene:'):
        scene_text = (line[7:].replace("]", "").replace(")", "").replace(":", ",").replace(".", ","))
        is_scene = True
    elif line.startswith('[Scene') or line.startswith('(Scene'):
        scene_text = (line[6:].replace("]", "").replace(")", "").replace(":", ",").replace(".", ","))
        is_scene = True
    elif line.startswith('['):
        scene_text = (line[1:].replace("]", "").replace(")", "").replace(":", ",").replace(".", ","))
        is_scene = True
    return scene_text, is_scene


def split_scene_to_location_and_decription(scene):
    if scene is None:
        return None, None
    # remove not informative commas
    if ',' in scene[:2]:
        scene = scene[2:]
    if ',' in scene[-2:]:
        scene = scene[:-2]

    if ',' in scene:
        spl = scene.split(',', 1)
        location = spl[0].lower().replace('’', "'").strip()
        scene_text = spl[1]
    elif ';' in scene:
        spl = scene.split(';', 1)
        location = spl[0].lower().replace('’', "'").strip()
        scene_text = spl[1]
    else:
        location = None
        scene_text = scene
    return scene_text, location


def addLineAccumulatorData(acc, speaker, text):
    word_count = len(text.split(" "))
    speaker = speaker.lower()
    addedToGender = False
    for girl in girls:
        if girl in speaker:
            acc[f'{girl}_sentence_count'] = acc.get(f'{girl}_sentence_count', 0) + 1
            acc[f'{girl}_word_count'] = acc.get(f'{girl}_word_count', 0) + word_count
            if addedToGender == False:
                acc['girls_sentence_count'] = acc.get('girls_sentence_count', 0) + 1
                acc['girls_word_count'] = acc.get('girls_word_count', 0) + word_count
                addedToGender = True
    for boy in boys:
        if boy in speaker:
            acc[f'{boy}_sentence_count'] = acc.get(f'{boy}_sentence_count', 0) + 1
            acc[f'{boy}_word_count'] = acc.get(f'{boy}_word_count', 0) + word_count
            if addedToGender == False:
                acc['boys_sentence_count'] = acc.get('boys_sentence_count', 0) + 1
                acc['boys_word_count'] = acc.get('boys_word_count', 0) + word_count
                addedToGender = True

    return acc


def unifyLocations(counter, froms, to):
    for from_loc in froms:
        if from_loc in counter.elements():
            counter[to] += counter[from_loc]
            del counter[from_loc]


def mergeVariationsCounter(counter):
    unifyLocations(counter, [
        "flashback scene",
        "flashback scene from last week"
    ], "flashback")

    unifyLocations(counter, [
        "central park",
        "cental perk",
        "ross is in central perk"
    ], "central perk")

    unifyLocations(counter, [
        "rachel and monica's",
        "rachel's bedroom",
        "monica and rachel's erm",
        "rachel's room",
        "monica's bedroom",
        "monica and rachel's",
        "monica",
        "monica and phoebe's",
        "monica's apartment",
        "monica and rachel's apartment",
        "chandler and monica's apartment",
        "monica and chandler's",
        "monica and chandler's bedroom",
        "chandler and monica's",
    ], "monica and chandler's apartment")
    unifyLocations(counter, [
        "chandler and joey's apartment",
        "chandler and joey's erm",
        "joey and rachel's apartment",
        "joey and rachel's",
        "joey's bedroom",
        "chandler's apartment",
        "chandler's",
        "chandler and joey's",
        "chandler and eddie's apartment"
        "at chandler and joey's",
    ], "joey's apartment")

    unifyLocations(counter, [
        "ross' apartment",
    ], "ross's apartment")

    unifyLocations(counter, [
        "phoebe's",
        "phoebe and rachel's",
    ], "phoebe's apartment")

    unifyLocations(counter, [
        "the hallway between the apartments",
        "cut to the hallway"
    ], "the hallway")

    unifyLocations(counter, [
        "ross and rachel's",
    ], "ross and rachel's apartment")

    unifyLocations(counter, [
        "restaurant",
    ], "a restaurant")
    return counter


def findLastLocation(locations):
    for loc in reversed(locations):
        if loc != None and loc != "time lapse" and loc != "cut to later":
            return loc
    return None


def extract_episode_name(ep_name):
    if len(ep_name) > 4:
        ep_name = ep_name[2:5] + ep_name[7:]
    else:
        ep_name = ep_name[2:]
    return ep_name


def attachCountingLocations(acc, counter):
    for loc in most_common_location:
        acc[f'{loc}_sentence_count'] = counter[loc]


def convert_script_to_df(season, season_dir_path, ep_name):
    unusable_prefixs = ['Written by:', 'Transcribed by:', 'Additional transcribing by:', '(Note:',
                        'With Minor Adjustments by:',
                        'With Help From:', 'With Minot Adjustments by:', '{Transcriber\'s Note:', 'Story by:',
                        'Teleplay by:',
                        'NOTE:']

    script_path = season_dir_path + f'/{ep_name}.txt'
    scene = None

    all_rows = {}
    all_rows['Scene'] = []
    all_rows['speaker'] = []
    all_rows['text'] = []

    accumulators = {}

    locationCnt = collections.Counter()

    with open(script_path, "r+") as fp:
        for cnt, line in enumerate(fp):
            if line == '\n':
                continue

            scene_text, is_scene = extract_scene_text(line)
            if is_scene:
                scene = scene_text
                continue

            if scene is not None:
                if ':' in line and all([not line.startswith(bad_prefix) for bad_prefix in unusable_prefixs]):
                    speaker = line[:line.index(':')]
                    text = line[line.index(':') + 1:].replace(":", "").replace('"', '').replace("'", "").replace(",",
                                                                                                                 "").replace(
                        "\n", "")
                    text = re.sub(r'\([^)]*\)', '', text)
                    text = re.sub(r'\[[^()]*\]', '', text)
                    all_rows['speaker'].append(speaker)
                    all_rows['Scene'].append(scene)
                    all_rows['text'].append(text)
                    addLineAccumulatorData(accumulators, speaker, text)

    scenes = []
    locations = []
    for scene in all_rows['Scene']:
        scene_text, location = split_scene_to_location_and_decription(scene)
        scenes.append(scene_text)
        if location == 'time lapse' or location == 'cut to later':
            location = findLastLocation(locations)
        locationCnt[location] += 1
        locations.append(location)

    ep_name = extract_episode_name(ep_name)
    episode_col = [ep_name] * len(scenes)
    season_col = [season] * len(scenes)
    df = pd.DataFrame(list(zip(season_col, episode_col, scenes, locations, all_rows['speaker'], all_rows['text'])),
                      columns=['Season', 'Episode', 'Scene', 'Location', 'Speaker', 'Text'])

    del locationCnt[None]
    if len(locationCnt.most_common(1)) > 0:
        mergeVariationsCounter(locationCnt)
        attachCountingLocations(accumulators, locationCnt)
        accumulators['most_sentences_in_location'] = locationCnt.most_common(1)[0][0]
    return df, accumulators


def initialCounters():
    row = {}
    row['season'] = []
    row['episode'] = []
    for speaker in boys + girls:
        row[f'{speaker}_sentence_count'] = []
        row[f'{speaker}_word_count'] = []
    for loc in most_common_location:
        row[f'{loc}_sentence_count'] = []
    row['girls_sentence_count'] = []
    row['boys_sentence_count'] = []
    row['girls_word_count'] = []
    row['boys_word_count'] = []
    row['most_sentences_in_location'] = []

    return row


def addToCounters(counters, ep_counters, episode, season):
    counters['episode'].append(extract_episode_name(episode))
    counters['season'].append(season)
    for key in counters.keys():
        if key not in ['episode', 'season']:
            counters[key].append(ep_counters.get(key, 0))
    return counters


def boysOrGirls_word(row):
    if row['girls_word_count'] > row['boys_word_count']:
        return 'girls'
    return 'boys'


def boysOrGirls_sentences(row):
    if row['girls_sentence_count'] > row['boys_sentence_count']:
        return 'girls'
    return 'boys'


def actor_most_sentences(row):
    most_sent_actor = None
    for actor in boys + girls:
        if (not most_sent_actor) or row[f'{actor}_sentence_count'] > row[f'{most_sent_actor}_sentence_count']:
            most_sent_actor = actor
    return most_sent_actor


def actor_most_words(row):
    most_sent_actor = None
    for actor in boys + girls:
        if (not most_sent_actor) or row[f'{actor}_word_count'] > row[f'{most_sent_actor}_word_count']:
            most_sent_actor = actor
    return most_sent_actor


def parse_text_scripts_to_csv():
    lines_df = []
    counters = initialCounters()

    for season in seasons:
        season_dir_path = 'data/season' + str(season)
        for episode in show_episodes_names[season - 1]:
            episode_df, episode_counters = convert_script_to_df(season, season_dir_path, episode)
            addToCounters(counters, episode_counters, episode, season)
            lines_df.append(episode_df)

    lines_df = pd.concat(lines_df)
    lines_df['Speaker'] = lines_df['Speaker'].str.lower()

    lines_df.to_csv('data/merged_lines.csv', index=False)

    counter_df = pd.DataFrame(counters)
    counter_df['gender_words_counter'] = counter_df.apply(boysOrGirls_word, axis=1)
    counter_df['gender_sentences_counter'] = counter_df.apply(boysOrGirls_sentences, axis=1)
    counter_df['actor_most_sentences'] = counter_df.apply(actor_most_sentences, axis=1)
    counter_df['actor_most_words'] = counter_df.apply(actor_most_words, axis=1)

    locationCnt = collections.Counter()

    for loc in lines_df['Location']:
        locationCnt[loc] += 1
    mergeVariationsCounter(locationCnt)

    location_df = pd.DataFrame()
    location_df['locations'] = locationCnt.keys()
    location_df['count'] = locationCnt.values()

    counter_df.to_csv('data/merged_counters.csv', index=False)
    location_df.to_csv('data/location_counters.csv', index=False)


def download_imdb_rating_data():
    imdb_crawler = IMDb()
    friends_id = '0108778'
    friends_series = imdb_crawler.get_movie(friends_id)
    imdb_crawler.update(friends_series, 'episodes')

    all_seasons_data = []

    for season in seasons:
        season_imdb = friends_series['episodes'][season]
        for episode_imdb in season_imdb.values():
            episode_id = episode_imdb.movieID
            episode_title = episode_imdb.data['title']
            episode = episode_imdb.data['episode']
            rating = episode_imdb.data['rating']
            votes = episode_imdb.data['votes']
            date = episode_imdb.data['original air date']
            plot = episode_imdb.data['plot']
            row = collections.OrderedDict()
            row['season'] = season
            row['episode'] = episode
            row['episode_id'] = episode_id
            row['episode_title'] = episode_title.replace('\n ', '')
            row['episode_rating'] = rating
            row['episode_votes'] = votes
            row['episode_date'] = date
            row['episode_plot'] = plot.replace('\n    ', '')
            all_seasons_data.append(row)
    df = pd.DataFrame(all_seasons_data)
    df.to_csv('data/friends_metadata.csv', index=False)


def combine_double_episodes_rating(metadata_df):
    # double_episodes = {2: 12, 6: 15, 9: 23}
    metadata_df['episode_rating'].iloc[35] = 8.7
    metadata_df.drop(index=36, inplace=True)
    metadata_df.drop(index=136, inplace=True)
    metadata_df['episode_rating'].iloc[214] = 8.55
    metadata_df.drop(index=217, inplace=True)
    metadata_df.drop(index=96, inplace=True)
    metadata_df.drop(index=120, inplace=True)
    metadata_df['episode_rating'].iloc[94] = 8.95
    metadata_df['episode_rating'].iloc[117] = 8.95
    metadata_df.drop(index=145, inplace=True)
    metadata_df['episode_rating'].iloc[140] = 9.15
    metadata_df.drop(index=193, inplace=True)
    metadata_df['episode_rating'].iloc[187] = 8.85

    return metadata_df


def create_corr_series(metadata_df, counters_df, season):
    counters_df = counters_df[
        ['season'] + [f'{entity}_sentence_count' for entity in most_common_location + boys + girls]]
    counters_df['ratings'] = list(metadata_df['episode_rating'])
    corr_df = counters_df.corr()['ratings']
    corr_df['season'] = season
    corr_df = corr_df[:-1]
    return corr_df


def normalize_counters(counters_df):
    from sklearn import preprocessing
    columns_to_normalize = [['ross_sentence_count', 'chandler_sentence_count', 'joey_sentence_count', 'rachel_sentence_count', 'monica_sentence_count', 'phoebe_sentence_count'],
                            [f'{entity}_sentence_count' for entity in most_common_location],
                            ['girls_sentence_count', 'boys_sentence_count']]
    for columns_group in columns_to_normalize:
        df = counters_df[columns_group]
        normalized_df = df.divide(pd.Series(df.sum(axis=1)), axis=0)
        counters_df[columns_group] = normalized_df
    return counters_df


def create_corr_tables():
    import warnings
    warnings.filterwarnings('ignore')

    counters_df = pd.read_csv('data/merged_counters.csv')
    counters_df = counters_df[
        ['season', 'episode', 'girls_sentence_count', 'boys_sentence_count'] + [f'{entity}_sentence_count' for entity in
                                                                                most_common_location + boys + girls]]
    counters_df = normalize_counters(counters_df)
    metadata_df = pd.read_csv('data/friends_metadata.csv')
    metadata_df = combine_double_episodes_rating(metadata_df)

    seasons_corr = []
    for season in seasons:
        season_metadata = metadata_df[metadata_df['season'] == season][['season', 'episode', 'episode_rating']]
        season_counters = counters_df[counters_df['season'] == season]
        season_corr = create_corr_series(season_metadata, season_counters, season)
        seasons_corr.append(season_corr)

    show_corr = create_corr_series(metadata_df, counters_df, 'All')
    seasons_corr.append(show_corr)

    df = pd.DataFrame(seasons_corr)
    columns = ['season'] + [f'{entity}_correlation' for entity in most_common_location + boys + girls]
    df.columns = columns
    df.to_csv('data/correlation_table.csv', index=False)


def main():
    print('downloading episodes')
    download_all_episodes()
    print('parsing scripts to csv files')
    parse_text_scripts_to_csv()
    print('downloading episodes ratings and metadata from imdb')
    download_imdb_rating_data()
    print('creating correlation data')
    create_corr_tables()


if __name__ == '__main__':
    main()