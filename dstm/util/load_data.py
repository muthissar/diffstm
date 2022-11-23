import pickle
import numpy as np
import torch
from pathlib import Path
import re
import pretty_midi
from functools import reduce
import urllib
import tarfile
import tqdm.contrib.concurrent
def loop_data(dataset, max_length):
    """[summary]

    Args:
        split (str): train/test/valid
        max_length (int): Filters and truncates to this size . Defaults to 512.

    Returns:
        [type]: [description]
    """
    # dataset = []
    # dataset = load_nes(split)
    n_samples = len(dataset)
    n_outs = dataset[0].shape[1]
    piano_rolls = np.empty((n_samples, max_length, n_outs), dtype="float32")

    def inner_loop(i):
        l = 0
        while True:
            # Augment data by looping until data is specified length
            for s in dataset[i]:
                # Use only channels 0, 1 (Pulse 1, Pulse 2)
                piano_rolls[i, l] = s
                l += 1
                if l == max_length:
                    return

    for i, _ in enumerate(dataset):
        inner_loop(i)
    return torch.Tensor(piano_rolls)


def load_nes(split):
    """Reads NES dataset

    Args:
        split (str): train/test/valid

    Returns:
        [type]: [description]
    """
    dataset = []
    for file in Path('data/nesmdb/nesmdb24_seprsco/{}/'.format(split)).rglob("*.pkl"):
        with open(file, "rb") as f:
            song = pickle.load(f)
            dataset.append(song[2])
    return dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def pin_memory(self):
        self.data = list(map(lambda x: x.pin_memory(), self.data))
        return self
    def __getitem__(self, index):
        return torch.Tensor(self.data[index])

    def __len__(self):
        return len(self.data)
    def get_longest_seq(self):
        return reduce(lambda a, b: a if a > int(b.shape[0]) else int(b.shape[0]), self.data, -1)

class PicklableSeqCollate:
    def __init__(self, seq_max_length) -> None:
        self.seq_max_length = seq_max_length
    def __call__(self, x):
        lengths = []
        y = len(x) * [None]
        for i, t in enumerate(x):
            seq_length = t.shape[0]
            if seq_length > self.seq_max_length:
                lengths.append(self.seq_max_length)
                #NOTE: randomly sample seq of max length
                end_index = np.random.randint(self.seq_max_length, seq_length)
                #x[i] = t[(end_index-seq_max_length):end_index, :]
                y[i] = t[(end_index-self.seq_max_length):end_index, :]
            else:
                lengths.append(seq_length)
                y[i] = t
        lengths = torch.tensor(lengths)
        #padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
        return padded, lengths
        

class DataPreprocessing:
    """Abstract class impliment the
    """

    def __init__(self, nb_classes, loop=False):
        self.nb_classes = nb_classes
        self.eye = np.eye(nb_classes)
        self.max_length = 1024
        self.loop = loop
        #self.dataset_size= dataset_size
        

    def preprocessed_cache(self, split, num_samples=None):
        folder = f'{self.folder}/cache'
        path = '{}/{}'.format(folder, split)
        path += ('_loop.pt' if self.loop else '.pt')
        #TODO: For now we store whole piano roll, though could be compressed by argmax
        if self.loop:
            raise Exception("Deprecated Parameter. Should be removed.")
        if Path(path).exists():
            dataset = torch.load(path)
            return dataset[:num_samples]
        else:
            print("Creating cache for {} data partion".format(split))
            Path(f'{folder}').mkdir(parents=True, exist_ok=True)
            dataset = self.get_encoded(split, num_samples)
            if self.loop:
                dataset = loop_data(dataset, self.max_length)
            torch.save(dataset, path)
        return dataset


    def get_data_loader(self, split, dataset=None, num_samples=None, seq_max_length=float('inf'), **kwargs):
        if dataset is None:
            dataset = self.preprocessed_cache(split, num_samples)
        if self.loop:
            collate_fn = None
        else:
            # def collate_fn(x):
            #     lengths = []
            #     y = len(x) * [None]
            #     for i, t in enumerate(x):
            #         seq_length = t.shape[0]
            #         if seq_length > seq_max_length:
            #             lengths.append(seq_max_length)
            #             #NOTE: randomly sample seq of max length
            #             end_index = np.random.randint(seq_max_length, seq_length)
            #             #x[i] = t[(end_index-seq_max_length):end_index, :]
            #             y[i] = t[(end_index-seq_max_length):end_index, :]
            #         else:
            #             lengths.append(seq_length)
            #             y[i] = t
            #     lengths = torch.tensor(lengths)
            #     #padded = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
            #     padded = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
            #     return padded, lengths
            collate_fn = PicklableSeqCollate(seq_max_length)
        cds = CustomDataset(dataset)
        data_loader = torch.utils.data.DataLoader(
            cds,
            collate_fn=collate_fn,
            #sampler = DistributedSampler(cds)
            **kwargs
        )
        return data_loader
    # TODO: this is only for midi with specified file_regex

    def masked_one_hot_encode(self, s, mask):
        encoded = self.one_hot_encode(s)
        return encoded[..., mask]

    def one_hot_encode(self, s):
        return self.eye[s]


class NesPreprocessing(DataPreprocessing):
    def __init__(self, **kwargs):
        self.folder = 'out/nes_mdb/data_cache'
        self.masks = [
            [i for i in range(0, 109) if i not in range(1, 32)],
            [i for i in range(0, 109) if i not in range(1, 32)],
            [i for i in range(0, 109) if i not in range(1, 21)],
            [i for i in range(0, 109) if i not in range(17, 109)]
        ]
        # self.nb_classes = 109
        self.d = len(self.masks[0])
        super().__init__(nb_classes=109, **kwargs)

    def get_encoded(self, split):
        data_list = load_nes(split)
        one_hot_encoded = []
        for sample in data_list:
            one_hot_encoded.append(
                self.masked_one_hot_encode(sample[:, 0], self.masks[0]))
            one_hot_encoded.append(
                self.masked_one_hot_encode(sample[:, 1], self.masks[1]))
        return one_hot_encoded


class EssenPreprocessing(DataPreprocessing):
    def __init__(self, **kwargs):
        self.folder = 'out/essen/data_cache'
        self.data_folder = 'data/essen_all'
        # todo: compute auto
        self.mask = [1,   2,   3,   4,   5,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
        92,  93,  94,  96, 255, 256]
        # self.nb_classes = 257
        self.d = len(self.mask)
        self.file_regex = re.compile(r'^([a-z]+)([0-9]+)\.mid$')
        super().__init__(nb_classes=257, **kwargs)

    def get_encoded(self, split):
        data_list = self.load_essen(split)
        one_hot_encoded = []
        for sample in data_list:
            one_hot_encoded.append(
                self.masked_one_hot_encode(sample, self.mask))
        return one_hot_encoded

    def load_essen_midi(path):
        pattern = midi.read_midifile(str(path))
        # Turn everything into 16th nodes
        track = pattern[0]
        res = pattern.resolution
        # ts = pattern[0][5]
        step_size = res // 4
        notes = []
        midinotes = track[8:-1]
        global_step = 0
        next_note = -1
        next_change = 0
        rest_number = 256
        can_shorten = False
        for event in midinotes:
            next_change = next_change + event.tick
            current_note = next_note
            if isinstance(event, midi.NoteOffEvent):
                next_note = rest_number
                # if next_change - global_step > step_size:
                #     can_shorten = True
                # else:
                #     can_storten = False
            else:
                next_note = event.data[0]
                note_on_time = global_step
            # Shorten merging note if possible
            # Does not work for 3 consecutive 16th notes as the mid is removed can_shorten
            if global_step < next_change and len(notes) > 1 and notes[-1] == current_note and notes[-2] == current_note:
                # silence
                notes[-1] = rest_number

            while global_step < next_change:
                notes.append(current_note)
                global_step += step_size
        return notes

    def load_essen(self, split):
        """Reads NES dataset

        Args:
            split (str): train/test/valid

        Returns:
            [type]: [description]
        """
        dataset = []
        for path in Path('{}/{}/'.format(self.data_folder, split)).glob("*.mid"):
        # for path in Path('data/essen_all/').rglob("*.mid"):
            dataset.append(EssenPreprocessing.load_essen_midi(path))
        return dataset


class SessionPreprocessing(DataPreprocessing):
    N_FILES = 45849
    def __init__(self, max_workers=10,**kwargs):
        self.folder = 'out/session'
        self.data_folder = 'data/session'
        self.max_workers=max_workers
        #self.file_regex = re.compile(r'^([a-z]+)([0-9]+)\.mid$')
        # todo: compute auto
        # self.mask = list(range(255))
        self.mask = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
            71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
        # self.nb_classes = 257
        self.d = len(self.mask) + 1
        #self.file_regex = re.compile(r'^sessiontune([0-9]+)\.mid$')
        #TODO: nb_classes is not really meaningfull any longer
        super().__init__(nb_classes=self.d, **kwargs)
    # def set_dataset_files(self):
    #     if self.dataset_size == "small":
    #         self.dataset_files = {'train': "train_6000.pt", "valid": "valid_600.pt", "test": "test_600.pt"}
    #     elif self.dataset_size == "medium":
    #          self.dataset_files = {'train': "train_15000.pt", "valid": "valid_1500.pt", "test": "test_1500.pt"}
    #     elif self.dataset_size == "large":
    #          self.dataset_files = {'train': "train_36679.pt", "valid": "valid_4585.pt", "test": "test_4585.pt"}
    #     else:
    #         raise NotImplementedError("Dataset size {} is not implimented.".format(self.dataset_size))
    def get_encoded(self, split, num_samples):
        data_list = self.load(split)
        one_hot_encoded = []
        for i, sample in enumerate(data_list):
            one_hot_encoded.append(sample)
            if num_samples is not None and i == num_samples-1:
                break
        return one_hot_encoded

    def load_midi(self, path):
        # TODO: for other dataset (change 120 to load from file)
        pm = pretty_midi.PrettyMIDI(path)
        pps = 120/60
        sixtheensthps = pps/(16)
        sixtheensthps
        piano_roll = pm.get_piano_roll(fs=1/sixtheensthps)
        masked = piano_roll.transpose()[:, self.mask]
        masked_with_rest = np.concatenate((np.zeros((masked.shape[0], 1)), masked), axis=1)
        #TODO: might be inefficient:
        #forces output to be unison (by sellecting only highest pitch)
        for i, outs in enumerate(masked_with_rest):
            highest_pitch = 0
            for j, pitch in enumerate(outs):
                if pitch != 0.0:
                    highest_pitch = j
                masked_with_rest[i, j] = 0.0
            masked_with_rest[i, highest_pitch] = 1.0
        return torch.FloatTensor(masked_with_rest)
    def get_files_split(self, split):
        return sorted(Path(f'{self.data_folder}/{split}/').rglob("*.mid"), key=lambda x: int(x.name[11:-4]))
    def load(self, split):
        """Reads midi dataset

        Args:
            split (str): train/test/valid

        Returns:
            [type]: [description]
        """
        #dataset = []
        # instead we return a generator
        #for path in Path('{}/{}/'.format(self.data_folder, split)).glob("*.mid"):           
        #    dataset.append(SessionPreprocessing.load_midi(str(path)))
        files = self.get_files_split(split)
        return tqdm.contrib.concurrent.thread_map(
            lambda path: self.load_midi(str(path)),
            files,
            max_workers=self.max_workers,
            chunksize=int(len(files)/(10*self.max_workers)))
    #def download_dataset(self):
        
    def prepare_dataset(self):
        #TODO: should be more robust (checksum)
        download_folder = Path(self.folder).parent.joinpath("download")
        if not download_folder.exists():
            download_folder.mkdir(parents=True, exist_ok=True)
        for compressed_archive, url, out_dir in [
            (f'{download_folder}/dataset.tar.gz', 
            'https://github.com/IraKorshunova/folk-rnn/raw/master/data/midi.tgz',
            'data'
            ),
            (f'{download_folder}/checkpoints.tar.gz',
            'https://drive.jku.at/ssf/s/readFile/share/44884/1121046473828436352/publicLink/checkpoints.tar.gz',
            f'{self.folder}/model'
            )]:
            if not Path(compressed_archive).exists():
                print(f"Downloading {compressed_archive} from CDN")
                urllib.request.urlretrieve(url, compressed_archive)
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                tar = tarfile.open(compressed_archive)
                print(f"Extracting {compressed_archive}")
                tar.extractall(out_dir)
        splits = ["train", "valid", "test"]
        if not np.all([Path(f'{self.folder}/cache/{split}.pt').exists() for split in splits]):
            self.create_data_split()
        for split in splits:
            self.preprocessed_cache(split)
    def get_path_titles(self):
        paths_sorted = [Path(f'{self.data_folder}/sessiontune{i}.mid') for i in range(SessionPreprocessing.N_FILES)]
        with open(f'{self.data_folder}/allabcwrepeats_parsed', 'rt', encoding='UTF-8') as f:
        #with open(f'{Path(self.folder).parent.joinpath("download")}/allabcwrepeats_parsed', 'rt', encoding='UTF-8') as f:
            str_ = f.read()
        # NOTE: remove empty str and the last file is not included in midi.
        abc_strs = str_.split('\n\n')[:-2]
        regex = re.compile('T:(.+)\n')
        titles = list(map(lambda abc_str: re.sub(r'[^\w\-_\. ]', '_', regex.search(abc_str)[1]), abc_strs))
        return paths_sorted, titles
    def create_data_split(self):
        print('Creating datasplit.')
        paths_sorted, titles = self.get_path_titles()
        assert len(paths_sorted) == len(titles), "Abc and midi files should match."
        data_dict = {}
        for title, path in zip(titles, paths_sorted):
            if title not in data_dict:
                data_dict[title] = []
            data_dict[title].append(path)
        

        data_dict_keys = list(data_dict.keys())
        np.random.default_rng(42).shuffle(data_dict_keys)
        #n_files = sum(map(len, data_dict))
        n_train = int(10/12 * SessionPreprocessing.N_FILES)
        n_valid = int(11/12 * SessionPreprocessing.N_FILES)
        #n_test = n_files - n_train - n_valid
        n_splits_acc = [n_train, n_valid, SessionPreprocessing.N_FILES]
        #n_splits = np.diff(n_splits_acc, prepend=0)
        n_splits = [38213, 3818, 3818]
        partition_keys = [
            [],
            [],
            []
        ]
        split = 0
        file_counter = 0
        for key in data_dict_keys:
            if file_counter > n_splits_acc[split]:
                split += 1
            partition_keys[split].append(key)
            file_counter += len(data_dict[key])
        folders = ['train', 'valid', 'test']
        # for split in folders:
        file_counts = []
        for split_str, split_keys, n_split in zip(folders, partition_keys, n_splits):
            split_folder = Path('{}/{}'.format(self.data_folder, split_str))
            if split_folder.exists():
                if len(list(split_folder.glob('**/*.mid'))) == n_split:
                    file_counts.append(n_split)
                    continue
                else:
                    raise Exception('Unexpected number of files.')
            else:
                split_folder.mkdir(parents=True, exist_ok=True)
            file_count = 0
            for key in split_keys:
                paths =  data_dict[key]
                folder = Path(f'{self.data_folder}/{split_str}/{key}')
                folder.mkdir()
                for path in paths:
                    p = Path(f"{folder}/{path.name}").resolve()
                    p.symlink_to(path.resolve())
                    file_count+=1
            file_counts.append(file_count)
        print('Train/Valid/Test file count: {}/{}/{}'.format(*file_counts))
