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
import os
import random
import math
# import snap
import itertools
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange
from sklearn import preprocessing
from joblib import Parallel, delayed


def get_playlist_list(num=50):
    data_dir = "/path/to/mpd/data"
    data_files = list(sorted(os.listdir(data_dir)))
    data_files = data_files[:num]
    playlist_list = []
    for file_number, data_file in enumerate(tqdm(data_files)):
        if not data_file.startswith("."):
            with open(data_dir + data_file) as file:
                data = json.load(file)
                playlists = data['playlists']
                playlist_list += playlists
    return playlist_list

def get_tracks(playlist):
    return list(sorted(track['track_uri']
                       for track in playlist['tracks']))


def get_track_uri_from_index(playlist, track_index):
    return playlist['tracks'][track_index]['track_uri']

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

        for track_index, track_uri in enumerate(track_list):
#             to have unique track id
            if track_uri in tracks_dict:
                increasing_track_id = tracks_dict[track_uri]
            else:
                increasing_track_id = next_track_id
                tracks_dict[track_uri] = next_track_id
                next_track_id += 1
                g.AddNode(increasing_track_id)
            if (playlist_id == increasing_playlist_id and track_index == dropped_track_index):
                dropped_track_id = increasing_track_id
                continue
            g.AddEdge(increasing_track_id, increasing_playlist_id)
    tracks_id_dict = dict(zip(tracks_dict.values(), tracks_dict.keys()))
    return g, dropped_track_id, tracks_dict, tracks_id_dict


def make_graph(playlist_list):
    """
    returns
    g graph
    tracks_dict keys are track_uri, values are track_ids
    tracks_id_dict keys are track_ids, values are track_uris
    """
    g = snap.TUNGraph.New()
    next_track_id = len(playlist_list)
    tracks_dict = {}
    for increasing_playlist_id in range(len(playlist_list)):
        g.AddNode(increasing_playlist_id)
        track_list = get_tracks(playlist_list[increasing_playlist_id])

        for track_index, track_uri in enumerate(track_list):
            if track_uri in tracks_dict:
                increasing_track_id = tracks_dict[track_uri]
            else:
                increasing_track_id = next_track_id
                tracks_dict[track_uri] = next_track_id
                next_track_id += 1
                g.AddNode(increasing_track_id)
            g.AddEdge(increasing_track_id, increasing_playlist_id)
    tracks_id_dict = dict(zip(tracks_dict.values(), tracks_dict.keys()))
    return g, tracks_dict, tracks_id_dict

def make_graph_dict(playlist_list):
    """
    returns
    g graph
    tracks_dict keys are track_uri, values are track_ids
    tracks_id_dict keys are track_ids, values are track_uris
    """
    g = {}
    next_track_id = len(playlist_list)
    tracks_dict = {}
    for increasing_playlist_id in range(len(playlist_list)):
        assert increasing_playlist_id not in g
        g[increasing_playlist_id] = set([])

        track_list = get_tracks(playlist_list[increasing_playlist_id])

        for track_index, track_uri in enumerate(track_list):
            if track_uri in tracks_dict:
                increasing_track_id = tracks_dict[track_uri]
            else:
                increasing_track_id = next_track_id
                tracks_dict[track_uri] = next_track_id
                next_track_id += 1
                assert increasing_track_id not in g
                g[increasing_track_id] = set([])
            g[increasing_track_id].add(increasing_playlist_id)
            g[increasing_playlist_id].add(increasing_track_id)
    tracks_id_dict = dict(zip(tracks_dict.values(), tracks_dict.keys()))
    return g, tracks_dict, tracks_id_dict

def make_tracks_dict(playlist_list):
    """
    returns
    tracks_dict keys are track_uri, values are track_ids
    tracks_id_dict keys are track_ids, values are track_uris
    """
    next_track_id = len(playlist_list)
    tracks_dict = {}
    for increasing_playlist_id in range(len(playlist_list)):

        track_list = get_tracks(playlist_list[increasing_playlist_id])

        for track_index, track_uri in enumerate(track_list):
            if track_uri in tracks_dict:
                increasing_track_id = tracks_dict[track_uri]
            else:
                increasing_track_id = next_track_id
                tracks_dict[track_uri] = next_track_id
                next_track_id += 1
    tracks_id_dict = dict(zip(tracks_dict.values(), tracks_dict.keys()))
    return tracks_dict, tracks_id_dict

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

def process_playlists(fn, playlist_list):
    num_playlist = len(playlist_list)
    all_true_ids = []
    all_suggested_ids = []
    for playlist_id in tnrange(num_playlist):
        true_id, suggested_ids = fn(playlist_list, playlist_id)
        if true_id is None:
            continue
        all_true_ids.append(true_id)
        all_suggested_ids.append(suggested_ids)

    return all_true_ids, all_suggested_ids

def process_playlists_parallel(fn, playlist_list):
    num_playlist = len(playlist_list)
    fn_with_playlist_list = functools.partial(fn, playlist_list)
    results = Parallel(n_jobs=4)(tqdm(
        (delayed(fn_with_playlist_list)(i)
         for i in range(num_playlist)), total=num_playlist
    ))

    for true_id, suggested_ids in results:
        if true_id is None:
            continue
        all_true_ids.append(true_id)
        all_suggested_ids.append(suggested_ids)
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


def dcg(gains):
    value = 0
    for index, gain in enumerate(gains):
        if index == 0:
            value += gain
        else:
            value += gain/math.log(1+index)
    return value


def ndcg(gains):
    ideal_gains = list(sorted(gains, reverse=True))
    ideal_dcg = dcg(ideal_gains)
    if ideal_dcg == 0:
        return 0.0
    real_dcg = dcg(gains)
    return real_dcg/ideal_dcg

def make_track_artist_dict(playlist_list):
    track_artist_dict = {}
    for playlist in playlist_list:
        for track in playlist['tracks']:
            track_artist_dict[track['track_uri']] = track['artist_uri']
    return track_artist_dict

def get_track_uri_artist(track_uri):
    return track_artist_dict.get(track_uri)

def is_same_artist(track_artist_dict, tracks_id_dict, true_index, other_index):
    true_track_artist = track_artist_dict[tracks_id_dict[true_index]]
    other_track_artist = track_artist_dict[tracks_id_dict[other_index]]
    return true_track_artist == other_track_artist


def artist_get_gain(playlist_list, tracks_id_dict):
    track_artist_dict = make_track_artist_dict(playlist_list)

    def real_artist_get_gain(true_index, indices):
        gains = []
        for index in indices:
            if index == true_index:
                gains.append(2)
            elif is_same_artist(track_artist_dict, tracks_id_dict, true_index, index):
                gains.append(1)
            else:
                gains.append(0)
        return gains
    return real_artist_get_gain

def binary_get_gain(true_index, indices):
    gains = []
    for index in indices:
        if index == true_index:
            gains.append(1)
        else:
            gains.append(0)
    return gains


def cal_ndcg(all_true_indices, all_indices, gain=binary_get_gain):
    """
    for artist:
    cal_ndcg(all_true_indices, all_indices, gain=artist_get_gain(playlist_list))
    """
    result = 0
    count = 0
    total = len(all_true_indices)
    for true_index, indices in tqdm(zip(all_true_indices, all_indices), total=total):
        if hasattr(indices, 'shape'):
            indices = indices[0,:]
        gains = gain(true_index, indices)
        result += ndcg(gains)
        count += 1
    return result / count

def cal_results(playlist_list, all_true_indices, all_indices):
    ap500 = cal_ave_precision(500, all_true_indices, all_indices)
    print('AP@500', ap500)
    bin_ndcg = cal_ndcg(all_true_indices, all_indices, gain=binary_get_gain)
    print('binary ndcg', bin_ndcg)
    tracks_dict, tracks_id_dict = make_tracks_dict(playlist_list)
    artist_ndcg = cal_ndcg(all_true_indices, all_indices, gain=artist_get_gain(playlist_list, tracks_id_dict))
    print('artist ndcg', artist_ndcg)
    return {'AP@500': ap500, 'binary ndcg': bin_ndcg, 'artist ndcg': artist_ndcg}
