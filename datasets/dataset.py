import os
import random
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ei -- emotional intensity
# st -- statement
# at -- attempt
# ac -- actor
# xx-xx-em-ei-st-at-ac
# 03-01-01-01-02-02-19
# 03-01-04-02-02-02-19

class RAVDESSDataset(Dataset):
    emotion_id2emotion = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    valid_emotions = ['neutral', 'happy', 'sad', 'angry']

    def __init__(self, root_dir, target_sr=16000, transform=None):
        self.root_dir = root_dir
        self.target_sr = target_sr
        self.transform = transform
        self.file_list = []
        self.basic_resampler = torchaudio.transforms.Resample(48000, target_sr)
        self.db = {} 

        # IDEA. Resample all audio files to target_sr initially

        for root, _, files in os.walk(root_dir):
            for f in files:
                if not f.endswith(".wav"):
                    continue

                path = os.path.join(root, f)
                parts = f.split('-')
                
                emotion = parts[2]
                if self.emotion_id2emotion[emotion] not in self.valid_emotions:
                    continue
                # intensity = parts[3]
                # if intensity == '02':
                #     continue
                statement = parts[4]
                actor = parts[6].replace('.wav', '')
                
                self.file_list.append({'path': path, 'emotion': emotion, 'actor': actor, 'statement': statement})
                
                if actor not in self.db: self.db[actor] = {}
                if emotion not in self.db[actor]: self.db[actor][emotion] = {}
                if statement not in self.db[actor][emotion]: self.db[actor][emotion][statement] = []
                self.db[actor][emotion][statement].append(path)

    def __len__(self):
        return len(self.file_list)
    
    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.target_sr:
            if sr == 48000:
                waveform = self.basic_resampler(waveform)
            else:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)
        
        if self.transform:
            waveform = self.transform(waveform)
        return waveform

    def __getitem__(self, idx):
        anchor_info = self.file_list[idx]
        anchor_path = anchor_info['path']
        actor_id = anchor_info['actor']
        emotion_id = int(anchor_info['emotion']) - 1
        statement_id = anchor_info['statement']
        
        neutral_emotion = '01'
        neutral_options = self.db[actor_id][neutral_emotion][statement_id]
        
        neutral_path = random.choice(neutral_options)
        anchor_wave = self._load_audio(anchor_path)
        neutral_wave = self._load_audio(neutral_path)

        # print(anchor_path, neutral_path, emotion_id)
        
        return anchor_wave, emotion_id, neutral_wave


def collate_fn(batch):
    """
    batch: list of tuples (anchor_wave, emotion_id, neutral_wave)
    aka batch = [(anch1, em_id1, neut1), (anch2, em_id2, neut2), ..., (anch4, em_id4, neut4)]
    """
    # for triple in batch:
    #     print(triple[0].shape)

    anchor_waves = [item[0].squeeze(0) for item in batch]
    anchor_padded = pad_sequence(anchor_waves, batch_first=True)
    anchor_lens = torch.tensor([wave.shape[-1] for wave in anchor_waves], dtype=torch.long)
    
    emotion_ids = [item[1] for item in batch]
    emotion_ids = torch.tensor(emotion_ids, dtype=torch.long)

    neutral_waves = [item[2].squeeze(0) for item in batch]
    neutral_padded = pad_sequence(neutral_waves, batch_first=True)
    neutral_lens = torch.tensor([wave.shape[-1] for wave in neutral_waves], dtype=torch.long)

    # [B, T_max] -> [B, 1, T_max] нужно ли?
    # anchor_padded = anchor_padded.unsqueeze(1)
    # neutral_padded = neutral_padded.unsqueeze(1)
    
    return {
        "anchor": anchor_padded,
        "anchor_lens": anchor_lens,
        "emotion_id": emotion_ids,
        "neutral": neutral_padded,
        "neutral_lens": neutral_lens,
    }


def main():
    dataset = RAVDESSDataset('archive')
    print(len(dataset))
    element = dataset[23]
    print(element)
    print(element[0].shape, element[0].min(), element[0].max())

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    batch = next(iter(dataloader))
    print(batch["anchor"].shape)
    print(batch["anchor_lens"])

if __name__ == '__main__':
    main()