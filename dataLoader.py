'''
DataLoader for training (with error handling for corrupt files)
'''

import glob
import numpy as np
import os
import random
import soundfile
import torch
from scipy import signal

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames

        # Load data & labels
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

        # 유효한 데이터 인덱스 리스트 생성
        self.valid_indices = []
        for idx in range(len(self.data_list)):
            file_path = self.data_list[idx]
            try:
                # 파일이 정상적으로 열리는지 확인
                with soundfile.SoundFile(file_path) as f:
                    pass
                self.valid_indices.append(idx)
            except Exception as e:
                print(f"Skipping corrupt file during initialization: {file_path}, Error: {e}")

    def __getitem__(self, index):
        # 유효한 인덱스로 매핑
        index = self.valid_indices[index]

        file_path = self.data_list[index]
        speaker_label = self.data_label[index]

        try:
            # Read the utterance and randomly select the segment
            audio, sr = soundfile.read(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            # 파일 읽기에 실패하면 예외 발생
            raise IOError(f"Error reading file {file_path}: {str(e)}")

        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio], axis=0)

        return torch.FloatTensor(audio[0]), speaker_label

    def __len__(self):
        return len(self.valid_indices)
