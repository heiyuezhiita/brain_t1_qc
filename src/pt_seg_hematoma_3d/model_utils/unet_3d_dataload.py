# 2022.03.14 dataload summary
import os
import pickle
import torchio as tio
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from pt_seg_hematoma_3d.model.unet_3d_loss import *
from pt_seg_hematoma_3d.model_utils.unet_3d_train_figure import *
from my_general_utils.common_function import match_file_mult_dir


class Hematoma3dDataset:
    def __init__(self, config, logger=None):
        """
        :param config: config dic
        """
        # set logger
        self.logger = logger

        # get information ----------------------------
        # info: load by img dir
        self.img_dir = None
        self.mask_dir = None
        self.pattern = None

        # please load this dic by fun: load_data_by_dir or load_data_by_cv_fold_file
        self.train_dic = None
        self.valid_dic = None
        self.pred_dic = None

        # get other info from config
        self.cf = config
        self.classes = self.cf['model']['output_channel']  # for one-hot

        # get augment transformer info
        self.train_trans_dic = self.cf['transformer'].get('train', None)
        self.valid_trans_dic = self.cf['transformer'].get('valid', None)
        self.predict_trans_dic = self.cf['transformer'].get('predict', None)

    def load_data_by_dir(self, img_dir, mask_dir=None, pattern=None, dataset_mode='train',
                         len_suffix=7):
        """
        Data can be saved as .nii.gz format
        Each multi-modality data must in sub-folder of img_dir, which each subject dir contains a modality data
        Mask can be multi-class
        :param img_dir: string, contains all sub-folders, which stores the image with corresponding modality
        :param mask_dir: string, contains .nii.gz data. If mode is 'train' or 'valid', this arg is required
        :param pattern: string, to match all data, using regular expression, e.g.: r'HEMA_\d{5}_\d'
                        if None, assume that the default sort within each folder (or sub-folder) is one-to-one
        :param dataset_mode: string, must in ['train', 'valid', 'predict']
        :param len_suffix: int, the last 'len_suffix' strings in file name will be removed as subj_name
                                default 7 can remove '.nii.gz'
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.pattern = pattern

        # load subdir of each modality
        subdir_list = [i.path for i in os.scandir(self.img_dir) if i.is_dir()]
        modality_num = len(subdir_list)

        assert modality_num != 0, \
            self._set_logging('error', 'Modality dir not find in image dir: {}'.format(self.img_dir))

        # load mask or not
        if self.mask_dir:
            dir_list = subdir_list + [self.mask_dir]
        else:
            dir_list = subdir_list

        # match data (multi-image and mask), ths last content in each file_list is mask
        # if pattern exists, using pattern to match all files, else match by file order
        # print('pattern: {}'.format(self.pattern))
        if self.pattern:
            match_dic = match_file_mult_dir(dir_list, pattern=self.pattern)
        else:
            # match NAME by mask (or last subdir),
            # THIS IS GETTING NAME, NOT PATH! DO NOT REMOVE THIS LINE OF CODE!
            file_name_list = [i.name[:-len_suffix] for i in os.scandir(subdir_list[-1])]  # -7 is removing '.nii.gz'
            file_name_list.sort()
            # match all dir (contains mask if exist)
            file_path_all_list = []
            for dir_i in dir_list:
                file_path_list = [i.path for i in os.scandir(dir_i)]
                file_path_list.sort()
                file_path_all_list.append(file_path_list)
            # get match dir
            assert len(set(map(len, file_path_all_list))) == 1,\
                self._set_logging('error', "files number not same in all folders")
            match_dic = {i[0]: list(i[1:]) for i in zip(file_name_list, *file_path_all_list)}

        # changing the storage structure (list -> dic)
        match_dic_cp = {}
        for n, i in enumerate(match_dic.keys()):
            if self.mask_dir:
                img_path = match_dic[i][:-1]  # -1 is mask
                if isinstance(match_dic[i][-1], str):
                    mask_path = [match_dic[i][-1]]  # should into this branch
                else:
                    mask_path = match_dic[i][-1]
            else:
                img_path = match_dic[i]  # -1 is mask
                mask_path = ''

            match_dic_cp[i] = {'img': img_path, 'mask': mask_path, 'label': '', 'name': i}
        match_dic = match_dic_cp  # replace it

        if dataset_mode == 'train':
            self.train_dic = match_dic
        elif dataset_mode == 'valid':
            self.valid_dic = match_dic
        elif dataset_mode == 'predict':
            self.pred_dic = match_dic

    def load_data_by_cv_fold_file(self, fold_file_path, valid_fold_n, is_only_setting_one_mode=False,
                                  dataset_mode='predict'):
        """
        :param fold_file_path: load fold files which is created by s2_get_cross_validation_fold_seg
        :param valid_fold_n: int 0~n or int_list [n1, n2...],
                     which number will be set to valid and others_test will be train
        :param is_only_setting_one_mode: bool, if False, will setting train and valid by cv file
                                               if True, will only setting dataset by arg 'dataset_mode'
        :param dataset_mode: str, same as which in load_data_by_dir, but only be used if is_only_setting_one_mode=True
        """

        # load fold_files.pkl
        with open(fold_file_path, 'rb') as f:
            fold_path_dic = pickle.load(f)

        # convert valid_fold_n to list
        if isinstance(valid_fold_n, int):
            valid_fold_n = [valid_fold_n]

        # set train and validation
        if is_only_setting_one_mode:
            assert dataset_mode in ['train', 'valid', 'predict'], \
                self._set_logging('error',
                                  f"dataset_mode must in ['train', 'valid', 'predict'], input is {dataset_mode}")

            if dataset_mode == 'train':
                train_dic = {k: v for k, v in fold_path_dic.items() if k not in valid_fold_n}
                self.train_dic = reduce(lambda x, y: {**x, **y}, train_dic.values())  # merge all train dic
            elif dataset_mode == 'valid':
                valid_dic = {k: v for k, v in fold_path_dic.items() if k in valid_fold_n}
                self.valid_dic = reduce(lambda x, y: {**x, **y}, valid_dic.values())  # merge all valid dic
            elif dataset_mode == 'predict':
                pred_dic = {k: v for k, v in fold_path_dic.items() if k in valid_fold_n}
                self.pred_dic = reduce(lambda x, y: {**x, **y}, pred_dic.values())  # merge all valid dic
        else:
            train_dic = {k: v for k, v in fold_path_dic.items() if k not in valid_fold_n}
            self.train_dic = reduce(lambda x, y: {**x, **y}, train_dic.values())  # merge all train dic

            valid_dic = {k: v for k, v in fold_path_dic.items() if k in valid_fold_n}
            self.valid_dic = reduce(lambda x, y: {**x, **y}, valid_dic.values())  # merge all valid dic

    def get_dataset(self, dataset_mode='train'):
        """
        must run load_cv_fold_files before run this function
        fold_path_dic
            - fold_k
                - subj_name
                    - {'img': [img_path1, ..., img_pathn], 'mask': [mask_path], 'label': int, 'name': str}

        :param dataset_mode: str, must in ['train', 'valid', 'predict']

        :return: dataset, which format is tio.SubjectsDataset
        """
        # get data dic
        data_dic = self._get_data_dic(dataset_mode=dataset_mode)

        if dataset_mode == 'train':
            transformer = self._get_custom_transform_by_dic(self.train_trans_dic)
        elif dataset_mode == 'valid':
            transformer = self._get_custom_transform_by_dic(self.valid_trans_dic)
        elif dataset_mode == 'predict':
            transformer = self._get_custom_transform_by_dic(self.predict_trans_dic)

        # get tio.SubjectsDataset
        dataset = self._get_tio_subject_dataset_by_match_dic(data_dic, transform=transformer)

        return dataset

    def get_sampler(self, weight_dic, dataset_mode='train'):
        """
        :param weight_dic: {label_1: weight_1, label_2: weight_2, ...}
        :param dataset_mode:
        :return: WeightedRandomSampler
        """
        # get data dic
        data_dic = self._get_data_dic(dataset_mode=dataset_mode)

        # get label list
        label_list = [data_dic[i]['label'] for i in data_dic.keys()]
        label_count = Counter(label_list)

        # get sampler weight
        assert label_count.keys() == weight_dic.keys(),\
            self._set_logging('error', 'Key of weight_dic in config: {} not match with label in cv files: {}!'.format(
                weight_dic, label_count
            ))

        balanced_weight_dic = {i: 1 / label_count[i] * weight_dic[i] for i in label_count.keys()}
        balanced_weight_ts = torch.tensor([balanced_weight_dic[i] for i in label_list])

        self._set_logging('info', 'Balance weight: {}'.format(balanced_weight_dic))

        # get sampler
        sampler = WeightedRandomSampler(balanced_weight_ts, len(balanced_weight_ts), replacement=True)

        return sampler

    def _get_data_dic(self, dataset_mode):
        assert dataset_mode in ['train', 'valid', 'predict'], \
            self._set_logging('error', f"dataset_mode must in ['train', 'valid', 'predict'], input is {dataset_mode}")

        if dataset_mode == 'train':
            data_dic = self.train_dic
        elif dataset_mode == 'valid':  # valid and predict
            data_dic = self.valid_dic
        elif dataset_mode == 'predict':
            data_dic = self.pred_dic

        return data_dic

    @staticmethod
    def _get_tio_subject_dataset_by_match_dic(match_dic, transform):
        """
        :param match_dic: {subj_name: [img1_path, img2_path... mask_path]}
        :return:
        """
        # check whether a mask exists
        mask_path_test = match_dic[list(match_dic.keys())[0]]['mask']

        # get tio.SubjectsDataset
        tio_subj_list = []
        if len(mask_path_test):
            for subj_i in match_dic.keys():
                tio_subj_list.append(tio.Subject(
                    img=tio.ScalarImage(match_dic[subj_i]['img']),  # load multi-modality to each channel
                    mask=tio.LabelMap(match_dic[subj_i]['mask']),
                    label=match_dic[subj_i]['label'],
                    name=match_dic[subj_i]['name']))
        else:  # input '' to tio.LabelMap() will get error
            print('No mask input!')
            for subj_i in match_dic.keys():
                tio_subj_list.append(tio.Subject(
                    img=tio.ScalarImage(match_dic[subj_i]['img']),  # load multi-modality to each channel
                    mask='',
                    label=match_dic[subj_i]['label'],
                    name=match_dic[subj_i]['name']))

        # get tio.SubjectsDataset
        dataset = tio.SubjectsDataset(tio_subj_list, transform=transform)

        return dataset

    def _get_custom_transform_by_dic(self, trans_dic):
        trans_list = []
        for trans_name in trans_dic:
            if trans_name == 'RandomFlip':
                trans_list.append(tio.RandomFlip(**trans_dic[trans_name]))
            elif trans_name == 'RandomAffine':
                trans_list.append(tio.RandomAffine(**trans_dic[trans_name]))
            elif trans_name == 'CropOrPad':
                trans_list.append(tio.CropOrPad(**trans_dic[trans_name]))
            elif trans_name == 'RandomElasticDeformation':
                trans_list.append(tio.RandomElasticDeformation(**trans_dic[trans_name]))
            elif trans_name == 'RandomBiasField':
                trans_list.append(tio.RandomBiasField(**trans_dic[trans_name]))
            elif trans_name == 'RandomNoise':
                trans_list.append(tio.RandomNoise(**trans_dic[trans_name]))
            elif trans_name == 'RandomGamma':
                trans_list.append(tio.RandomGamma(**trans_dic[trans_name]))
            elif trans_name == 'OneHot':
                trans_list.append(tio.OneHot(**trans_dic[trans_name]))
            elif trans_name == 'Resample':
                trans_list.append(tio.Resample(**trans_dic[trans_name]))
            elif trans_name == 'Resize':
                trans_list.append(tio.Resize(**trans_dic[trans_name]))
            else:
                raise RuntimeError(self._set_logging('error', 'Unsupport transform name: {}'.format(trans_name)))

        self._set_logging('info', f"transform: {trans_list}")

        return tio.Compose(trans_list)

    def _set_logging(self, level, msg):
        if self.logger is not None:
            if level == 'debug':
                self.logger.debug(msg=msg)
            elif level == 'info':
                self.logger.info(msg=msg)
            elif level == 'warning':
                self.logger.warning(msg=msg)
            elif level == 'error':
                self.logger.error(msg=msg)
            else:
                self.logger.error('ERROR setting LEVEL: {}!\n message: {}'.format(level, msg))

        return msg