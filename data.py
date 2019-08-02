import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from random import sample
from typing import List, Sequence

import librosa as lr
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm


class AudioHelper():
    @staticmethod
    def webm_to_tensor(abspath: str) -> torch.Tensor:
        y, sr = lr.load(abspath, sr=16000)

        return AudioHelper._time_series_to_meltensor(y, sr)

    @staticmethod
    def file_to_tensor(abspath: str) -> torch.Tensor:
        offset = (int)(lr.get_duration(filename=abspath) / 2 - 1)
        y, sr = lr.load(abspath, sr=16000, duration=1, offset=offset)

        return AudioHelper._time_series_to_meltensor(y, sr)

    @staticmethod
    def _time_series_to_meltensor(y: np.ndarray,
                                  sr: int = 16000) -> torch.Tensor:
        lr.util.fix_length(y, sr)

        melspec = lr.feature.melspectrogram(y=y,
                                            sr=sr,
                                            power=2,
                                            n_fft=1024,
                                            n_mels=80)

        # +80 / 80 for 0..1
        mel_tensor = torch.from_numpy(
            ((lr.power_to_db(melspec, ref=np.max) + 40) / 40).T)

        return mel_tensor.unsqueeze(0).float()


class CsvHelper():
    # ..some..folders ( \ or / ) speakerId - workId - fileId
    _regex = re.compile(r"^.*(?:\\|\/)(\d+)-(?:\d+)-(?:\d+).flac$")

    @classmethod
    @lru_cache(1)
    def _filePathToId(cls, path: str) -> int:
        m = cls._regex.match(path)
        return int(m.groups()[0])

    @classmethod
    def create_test_csv(cls, path: str,
                        create_file: bool = True) -> pd.DataFrame:
        files = [
            os.path.abspath(fn)
            for fn in glob.iglob(f"{path}/**/*.flac", recursive=True)
        ]
        data_pd = cls._df_from_labels(files)
        if create_file:
            data_pd.to_csv(f"{path}/pairs.csv")
        return data_pd

    @classmethod
    def create_train_valid_csvs(cls,
                                path: str,
                                valid_percentage: int = 15,
                                create_file: bool = True
                                ) -> (pd.DataFrame, pd.DataFrame):
        files = [
            os.path.abspath(fn)
            for fn in glob.iglob(f"{path}/**/*.flac", recursive=True)
        ]

        labels = list({cls._filePathToId(fp) for fp in files})

        train_ids, valid_ids = cls._shuffle_split(labels, valid_percentage)
        valid_df = cls._df_from_labels(files, valid_ids)
        train_df = cls._df_from_labels(files, train_ids)

        if create_file:
            train_df.to_csv(f"{path}/train_pairs.csv")
            valid_df.to_csv(f"{path}/valid_pairs.csv")

        return (train_df, valid_df)

    @classmethod
    def _df_from_labels(cls,
                        files: Sequence[str],
                        included_labels: Sequence[int] = None) -> pd.DataFrame:

        # no walrus operator yet :=, so we use lru caching
        data_arr = np.asarray([(file, cls._filePathToId(file))
                               for file in files if included_labels == None
                               or cls._filePathToId(file) in included_labels])

        df = pd.DataFrame(data_arr, columns=['filename', 'label'])

        df["negative"] = df.apply(
            lambda x: df[df["label"] != x["label"]].sample(1)["filename"].item(
            ),
            axis=1,
        )
        df["positive"] = df.apply(
            lambda x: df[(df["label"] == x["label"]) & (df["filename"] != x[
                "filename"])].sample(1)["filename"].item(),
            axis=1,
        )

        df1 = df[["filename", "positive"]].copy()
        df1.columns = ["filename", "pair"]
        df1["label"] = 1

        df2 = df[["filename", "negative"]].copy()
        df2.columns = ["filename", "pair"]
        df2["label"] = 0

        return pd.concat([df1, df2])

    @staticmethod
    def _shuffle_split(labels: List[int],
                       percentage: int) -> (List[int], List[int]):

        labels = sample(labels, len(labels))
        split = int(np.floor(percentage / 100 * len(labels)))
        train_ids, valid_ids = labels[split:], labels[:split]
        return train_ids, valid_ids


class LibirSet(Dataset):
    def __init__(self, df: pd.DataFrame, read_workers: int = 8):
        self.data_pd = df
        self.data_len = len(self.data_pd)

        self.files_tensors = {}
        print("Reading files")

        def get_name_and_tensor(fname: str) -> (str, torch.Tensor):
            return fname, AudioHelper.file_to_tensor(fname)

        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            for fname, tens in tqdm(executor.map(
                    get_name_and_tensor, self.data_pd.filename.unique()),
                                    total=self.data_len // 2):
                self.files_tensors[fname] = tens

    @classmethod
    def from_csv(cls, csv_path: str, read_workers: int = 8):
        df = pd.read_csv(csv_path)
        return cls(df, read_workers)

    def __getitem__(self, index):
        row = self.data_pd.iloc[index]

        tensorA = self.files_tensors[row["filename"]]
        tensorB = self.files_tensors[row["pair"]]

        label = row["label"]

        return ((tensorA, tensorB), label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    # Call dataset
    print("Testing..")
    df = CsvHelper.create_test_csv(path=r".\LibriSpeech\test-clean")
    # libri_valid_dev = LibirSet.from_csv(
    #     r"P:\ml\ossr\LibriSpeech\dev-clean\valid_pairs.csv")
    libir_test = LibirSet(df)
    from pickle import HIGHEST_PROTOCOL
    torch.save(libir_test, "test_clean.pt", pickle_protocol=HIGHEST_PROTOCOL)
    # libri_dev = torch.load("dev_train.pt")
    # res = libri_dev[0]
    # print(res[0][0].shape)
