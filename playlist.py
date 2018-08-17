import multiprocessing
import functools
import json
import numpy as np
import sklearn
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import snap
import itertools
from tqdm import tqdm_notebook as tqdm
from sklearn import preprocessing


def get_playlist_list():
    filename = "../../mpd.slice.0-999.json"
    with open(filename) as file:
        data = json.load(file)
    playlist_list = []
    for i in range(len(data['playlists'])):
        playlist_list.append(data['playlists'][i])
    return playlist_list


def get_tracks(playlist):
    track_list = set([])
    for i in range(playlist['num_tracks']):
        track_list.add(playlist['tracks'][i]['track_uri'])
    return track_list


# A graph is made in which an edge between playlist-id and track-id is missing
# the aim is to predict the missing track in the playlist

# g, dropped_track_id = make_graph_drop_edge(playlist_list, playlist_id, dropped_track_index)/

# This function creats the track ids

def make_graph_drop_edge(playlist_list, playlist_id, dropped_track_index):
# What if a track was common between two playlist? If there is no dictionary, track+id +=1 produces
# 2 node ids for the same track.
    g = snap.TUNGraph.New()
    next_track_id = len(playlist_list)
    tracks_dict = {}
    dropped_track_id = None
    for increasing_playlist_id in range(len(playlist_list)):
        g.AddNode(increasing_playlist_id)
        track_list = get_tracks(playlist_list[increasing_playlist_id])
        track_list = sorted(track_list)
        
        for track_uri in track_list:
            track_index = track_list.index(track_uri)
#             to have unique track id
            if track_uri in tracks_dict:
                increasing_track_id = tracks_dict[track_uri]
            else:
                tracks_dict[track_uri] = next_track_id
                next_track_id += 1
                increasing_track_id = tracks_dict[track_uri]
                g.AddNode(increasing_track_id)
            if (playlist_id == increasing_playlist_id and track_index == dropped_track_index):
                dropped_track_id = increasing_track_id
                continue
            g.AddEdge(increasing_track_id, increasing_playlist_id)
    return g, dropped_track_id


# all_true_indices : list of indices that I dropped
# all_indices: list of list of best suggested track ids


# evaluation metric
def cal_ave_precision(k, all_true_indices, all_indices):
    sum_precision = 0.0
    for i in range(len(all_true_indices)):
        precision = float(all_true_indices[i] in all_indices[i][:k])
        sum_precision += precision
    average_precision = sum_precision / len(all_true_indices)
    return average_precision


def process_playlists(fn, playlist_list, multiprocess=True):
    if multiprocess:
        pool = multiprocessing.Pool(3)
    else:
        pool = None

    try:
        num_playlist = len(playlist_list)
        fn_with_playlist_list = functools.partial(fn, playlist_list)
        all_true_ids = []
        all_suggested_ids = []
    #     map(fn, [1,2,3,4]) -> [fn(1), fn(2), fn(3), fn(4)]
        if pool is not None:
            result = pool.imap_unordered(fn_with_playlist_list,
                                         range(num_playlist),
                                         chunksize=10)
        else:
            result = itertools.imap(fn_with_playlist_list,
                                    range(num_playlist))
        for true_id, suggested_ids in tqdm(result, total=num_playlist):
            if true_id is None:
                continue
            all_true_ids.append(true_id)
            all_suggested_ids.append(suggested_ids)
    except KeyboardInterrupt:
        if pool is not None:
            print 'TERMINATING POOL'
            pool.close()
            pool.join()
            pool.terminate()
    return all_true_ids, all_suggested_ids


def normalize_dataframe(dataframe):
    x = dataframe.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    new_dataframe = pd.DataFrame(x_scaled)
    new_dataframe.index = dataframe.index
    dataframe = new_dataframe
    return dataframe

def get_track_embedding(track_uri):
    return dataframe.loc[track_uri].values