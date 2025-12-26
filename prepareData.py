import os
import argparse
import configparser

import numpy as np
import pandas as pd


def seq2instance(data, num_his, num_pred):
    """
    data: [T, F]
    return:
        x: [num_sample, num_his, F]
        y: [num_sample, num_pred, F]
    """
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape=(num_sample, num_his, dims), dtype=float)
    y = np.zeros(shape=(num_sample, num_pred, dims), dtype=float)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y


def seq2instance_plus(data, num_his, num_pred):
    """
    data: [T, N, C]
    只用第一个通道作为输入/预测目标（流量）。
    return:
        x: [num_sample, num_his, N, 1]
        y: [num_sample, num_pred, N, 1]
    """
    num_step = data.shape[0]
    num_sample = num_step - num_his - num_pred + 1
    x_list, y_list = [], []
    for i in range(num_sample):
        x_list.append(data[i: i + num_his, :, :1])                        # [num_his, N, 1]
        y_list.append(data[i + num_his: i + num_his + num_pred, :, :1])   # [num_pred, N, 1]
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return x, y


def build_time_features_from_index(T, time_slice_size):
    """
    对没有时间戳的数据集（如 PEMS03 / PEMS07），
    根据索引构造 dayofweek / timeofday 特征。
    返回:
        time_feat: [T, 2]，列0为 weekday(0-6)，列1为 time-of-day(0~slots_per_day-1)
    """
    t_idx = np.arange(T)  # [0, 1, ..., T-1]
    num_slots_per_day = int(1440 // time_slice_size)

    dayofweek = (t_idx // num_slots_per_day) % 7
    timeofday = t_idx % num_slots_per_day

    dayofweek = dayofweek.reshape(-1, 1)
    timeofday = timeofday.reshape(-1, 1)

    time_feat = np.concatenate([dayofweek, timeofday], axis=-1)  # [T, 2]
    return time_feat


def build_time_features_from_timestamp(timestamps, time_slice_size):
    """
    对有时间戳的数据集（如 JiNan / PEMS04 / PEMS08），
    直接用最后一维的 UNIX 时间戳构造 dayofweek / timeofday。
    timestamps: [T]，单位秒
    返回:
        time_feat: [T, 2]
    """
    time_index = pd.to_datetime(timestamps, unit="s")
    time_index = pd.DatetimeIndex(time_index)

    dayofweek = np.reshape(time_index.weekday, (-1, 1))  # 0~6

    seconds_in_day = (
        time_index.hour * 3600
        + time_index.minute * 60
        + time_index.second
    )
    timeofday = seconds_in_day // (time_slice_size * 60)  # 一个时间片的索引
    timeofday = np.reshape(timeofday, (-1, 1))

    time_feat = np.concatenate([dayofweek, timeofday], axis=-1)  # [T, 2]
    return time_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="conf/JiNan_construct_samples.conf",
        type=str,
        help="configuration file path",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    print("Read configuration file: %s" % (args.config))
    config.read(args.config)

    data_config = config["Data"]
    training_config = config["Training"]

    # === 读取基础配置 ===
    time_slice_size = int(data_config["time_slice_size"])
    train_ratio = float(data_config["train_ratio"])
    val_ratio = float(data_config["val_ratio"])
    test_ratio = float(data_config["test_ratio"])

    num_his = int(training_config["num_his"])
    num_pred = int(training_config["num_pred"])

    data_file = data_config["data_file"]
    output_dir = data_config["output_dir"]



    print(f"Load data from: {data_file}")
    files = np.load(data_file, allow_pickle=True)
    data = files["data"]  # [T, N, C]
    print("data.shape =", data.shape)

    slices = data.shape[0]
    train_slices = int(slices * train_ratio)
    val_slices = int(slices * val_ratio)
    test_slices = slices - train_slices - val_slices

    train_set = data[:train_slices]
    val_set = data[train_slices: train_slices + val_slices]
    test_set = data[-test_slices:]

    sets = {"train": train_set, "val": val_set, "test": test_set}
    xy = {}
    te = {}

    for set_name, data_set in sets.items():
        print(f"\nProcessing set: {set_name}")
        print("  data_set.shape =", data_set.shape)

        # 1) X, Y：只用第一个通道（流量）
        X, Y = seq2instance_plus(
            data_set.astype("float64"), num_his, num_pred
        )
        xy[set_name] = [X, Y]

        # 2) 时间特征构造
        T = data_set.shape[0]

        if use_constructed_time == 1:

                print(
                    "  WARNING: data has no timestamp channel, "
                    "fallback to constructed time features from index."
                )
                time_feat = build_time_features_from_index(T, time_slice_size)

        # time_feat: [T, 2]
        te_h, te_y = seq2instance(time_feat, num_his, num_pred)
        # 拼成 [num_sample, num_his + num_pred, 2]
        te[set_name] = np.concatenate([te_h, te_y], axis=1).astype(np.int32)

        print("  X.shape =", X.shape, "Y.shape =", Y.shape)
        print("  TE.shape =", te[set_name].shape)

    x_trains, y_trains = xy["train"]
    x_vals, y_vals = xy["val"]
    x_tests, y_tests = xy["test"]
    trainTEs = te["train"]
    valTEs = te["val"]
    testTEs = te["test"]

    print("\n==== Final shapes ====")
    print("train:", x_trains.shape, y_trains.shape)
    print("val:", x_vals.shape, y_vals.shape)
    print("test:", x_tests.shape, y_tests.shape)
    print("trainTE:", trainTEs.shape)
    print("valTE:", valTEs.shape)
    print("testTE:", testTEs.shape)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir,
        f"samples_{num_his}_{num_pred}_{time_slice_size}.npz",
    )
    print(f"\nSave file to: {output_path}")


    print("Done.")
