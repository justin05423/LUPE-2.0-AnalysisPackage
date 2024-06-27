import numpy as np
import pandas as pd
from numba import jit
from numba.typed import List
from tqdm import tqdm
from utils.meta import *

def filter_pose_noise(pose, idx_selected, idx_llh, llh_value):
    datax = np.array(pose.iloc[:, idx_selected[::2]])
    datay = np.array(pose.iloc[:, idx_selected[1::2]])
    data_lh = np.array(pose.iloc[:, idx_llh])
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in range(data_lh.shape[1]):
        # TODO: load from config.ini the llh threshold
        llh = llh_value
        data_lh_float = data_lh[:, x].astype(float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(float)
    return currdf_filt, perc_rect


@jit(nopython=True)
def fast_standardize(data):
    a_ = (data - np.mean(data)) / np.std(data)
    return a_


def fast_nchoose2(n, k):
    a = np.ones((k, n - k + 1), dtype=int)
    a[0] = np.arange(n - k + 1)
    for j in range(1, k):
        reps = (n - k + j) - a[j - 1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1 - reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
        return a


@jit(nopython=True)
def fast_running_mean(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@jit(nopython=True)
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)


@jit(nopython=True)
def angle_between(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


@jit(nopython=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


@jit(nopython=True)
def fast_displacment(data, reduce=False):
    data_length = data.shape[0]
    if reduce:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 10)), dtype=np.float64)
    else:
        displacement_array = np.zeros((data_length, int(data.shape[1] / 2)), dtype=np.float64)
    for r in range(data_length):
        if r < data_length - 1:
            if reduce:
                count = 0
                for c in range(int(data.shape[1] / 2 - 2), data.shape[1], int(data.shape[1] / 2)):
                    displacement_array[r, count] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
                    count += 1
            else:
                for c in range(0, data.shape[1], 2):
                    displacement_array[r, int(c / 2)] = np.linalg.norm(data[r + 1, c:c + 2] - data[r, c:c + 2])
    return displacement_array


@jit(nopython=True)
def fast_length_angle(data, index):
    data_length = data.shape[0]
    length_2d_array = np.zeros((data_length, index.shape[1], 2), dtype=np.float64)
    for r in range(data_length):
        for i in range(index.shape[1]):
            ref = index[0, i]
            target = index[1, i]
            length_2d_array[r, i, :] = data[r, ref:ref + 2] - data[r, target:target + 2]
    length_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    angle_array = np.zeros((data_length, length_2d_array.shape[1]), dtype=np.float64)
    for k in range(length_2d_array.shape[1]):
        for kk in range(data_length):
            length_array[kk, k] = np.linalg.norm(length_2d_array[kk, k, :])
            if kk < data_length - 1:
                try:
                    angle_array[kk, k] = np.rad2deg(
                        angle_between(length_2d_array[kk, k, :], length_2d_array[kk + 1, k, :]))
                except:
                    pass
    return length_array, angle_array


@jit(nopython=True)
def fast_smooth(data, n):
    data_boxcar_avg = np.zeros((data.shape[0], data.shape[1]))
    for body_part in range(data.shape[1]):
        data_boxcar_avg[:, body_part] = fast_running_mean(data[:, body_part], n)
    return data_boxcar_avg


@jit(nopython=True)
def fast_feature_extraction(data, index):
    features = List()
    for n in range(len(data)):
        displacement_raw = fast_displacment(data[n])
        length_raw, angle_raw = fast_length_angle(data[n], index)
        features.append(np.hstack((length_raw[:, :], angle_raw[:, :], displacement_raw[:, :])))
    return features


@jit(nopython=True)
def fast_feature_binning(features, framerate, index):
    binned_features_list = List()
    for n in range(len(features)):
        bin_width = int(framerate / 10)
        for s in range(bin_width):
            binned_features = np.zeros((int(features[n].shape[0] / bin_width), features[n].shape[1]),
                                       dtype=np.float64)
            for b in range(bin_width + s, features[n].shape[0], bin_width):
                binned_features[int(b / bin_width) - 1, 0:index.shape[1]] = np_mean(features[n][(b - bin_width):b,
                                                                                    0:index.shape[1]], 0)
                binned_features[int(b / bin_width) - 1, index.shape[1]:] = np.sum(features[n][(b - bin_width):b,
                                                                                  index.shape[1]:], axis=0)
            binned_features_list.append(binned_features)
    return binned_features_list


def bsoid_extract_numba(data, fps):
    smooth = False
    index = fast_nchoose2(int(data[0].shape[1] / 2), 2)
    features = fast_feature_extraction(data, index * 2)
    binned_features = fast_feature_binning(features, fps, index * 2)
    return binned_features


def feature_extraction(train_datalist, num_train, framerate):
    f_integrated = []
    for i in tqdm(range(num_train)):
        data_list = List()
        data_list.append(train_datalist[i])
        binned_features = bsoid_extract_numba(data_list, framerate)
        f_integrated.append(binned_features[0])  # getting only the non-shifted
    return f_integrated


def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())
    return moving_avg


def get_avg_kinematics(predict, pose, bodypart, framerate=10, pixel_cm=pixel_cm):
    """

    :param predict:
    :param pose:
    :param bodypart:
    :param framerate:
    :param pixel_cm:
    :return: in centimeters
    """
    pose_estimate = pose[:, bodypart * 2:bodypart * 2 + 2]
    group_start = [0]
    group_start = np.hstack((group_start, np.where(np.diff(predict) != 0)[0] + 1))
    group_end = [len(predict) - 1]
    group_end = np.hstack((np.where(np.diff(predict) != 0)[0], group_end))

    bout_i_index = [np.arange(group_start[i], group_end[i]+1) for i in range(len(group_start))]
    # limiting to just the behavior bodypart x,y (10Hz) from start to end
    bout_pose_bodypart_i = [pose_estimate[bout_i_index[i], :] for i in range(len(bout_i_index))]

    behavior = predict[group_start]
    behavior_duration = np.hstack((np.diff(group_start), len(predict) - group_start[-1] + 1)) / framerate
    behavioral_start_time = group_start / framerate
    # kinematics portion
    bout_disp_all = {key: [] for key in behavior_names}
    bout_duration_all = {key: [] for key in behavior_names}
    bout_avg_speed_all = {key: [] for key in behavior_names}
    for b, behav in enumerate(behavior_names):
        behavior_j_bodypart_i_pose = []
        behavior_j_bout_duration = []
        behavior_index = np.where(behavior == b)[0]
        for instance in range(len(behavior_index)):
            # if behavior_duration[behavior_index][instance] > .1:
            behavior_j_bodypart_i_pose.append(bout_pose_bodypart_i[behavior_index[instance]])
            behavior_j_bout_duration.append(behavior_duration[behavior_index][instance])
        bout_avg_speed = []
        bout_duration = []
        bout_disp = []
        for n in range(len(behavior_j_bodypart_i_pose)):
            data_n_len = len(behavior_j_bodypart_i_pose[n])
            disp_list = []
            # going through the duration of each bout that's over 0.1s
            for r in range(data_n_len):
                if r < data_n_len - 1:
                    disp = []
                    # going through x and y
                    for c in range(0, behavior_j_bodypart_i_pose[n].shape[1], 2):
                        disp.append(
                            np.linalg.norm(behavior_j_bodypart_i_pose[n][r + 1, c:c + 2] -
                                           behavior_j_bodypart_i_pose[n][r, c:c + 2])*pixel_cm)
                    disp_list.append(disp)
            disp_r = np.array(disp_list)
            disp_feat = np.array(disp_r)
            bout_disp.append(np.mean(disp_feat)*framerate)
            bout_duration.append(behavior_j_bout_duration[n])
            bout_avg_speed.append(np.sum(disp_feat) / behavior_j_bout_duration[n])
        bout_disp_all[behav] = bout_disp
        bout_duration_all[behav] = bout_duration
        bout_avg_speed_all[behav] = bout_avg_speed

    return behavior, behavioral_start_time, behavior_duration, bout_disp_all, bout_duration_all, bout_avg_speed_all


def weighted_smoothing(predictions, size):
    predictions_new = predictions.copy()
    group_start = [0]
    group_start = np.hstack((group_start, np.where(np.diff(predictions) != 0)[0] + 1))
    for i in range(len(group_start) - 3):
        # sandwich jitters within a bout (jitter size defined by size)
        if group_start[i + 2] - group_start[i + 1] < size:
            if predictions_new[group_start[i + 2]] == predictions_new[group_start[i]] and \
                    predictions_new[group_start[i]:group_start[i + 1]].shape[0] >= size and \
                    predictions_new[group_start[i + 2]:group_start[i + 3]].shape[0] >= size:
                predictions_new[group_start[i]:group_start[i + 2]] = predictions_new[group_start[i]]

    for i in range(len(group_start) - 3):
        # replace jitter by previous behavior when it does not reach size
        if group_start[i + 1] - group_start[i] < size:
            predictions_new[group_start[i]:group_start[i + 1]] = predictions_new[group_start[i] - 1]
    return predictions_new