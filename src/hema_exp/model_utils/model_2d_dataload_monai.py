# 2023.01.19 dataload summary
# 2024.12.18 It was disassembled from the original "Hematoma2dDataset" in order to remove the strong dependency on "tio"
# In addition, a new dataloader based on MONAI will be built on other scripts
import monai
from hema_exp.model_utils.model_dataloader_base import ImageBaseDataset


# copy from Hematoma3dDataset
# edit in 2023.02.05, load data by dir can add label file (function: load_data_by_dir)
# edit in 2023.02.12, adding arg "ignore_fold" when using "cv" mode to load data (function: load_data_by_cv_fold_file)
# class name is 2dDataset, but load 5d torchio data(b, c, d1, d2, d3)
class ImageDatasetBasedOnMONAI(ImageBaseDataset):
    def __init__(self, config, logger=None):
        """
        :param config: config dic
        """
        super(ImageDatasetBasedOnMONAI, self).__init__(config, logger=logger)

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
            transformer = self._get_custom_monai_transform_by_dic(self.train_trans_dic)
        elif dataset_mode == 'valid':
            transformer = self._get_custom_monai_transform_by_dic(self.valid_trans_dic)
        elif dataset_mode == 'predict':
            transformer = self._get_custom_monai_transform_by_dic(self.predict_trans_dic)

        # get tio.SubjectsDataset
        dataset = self._get_monai_subject_dataset_by_match_dic(data_dic, transform=transformer)
        self._set_logging('info', f"dataset num: {len(dataset)}")

        return dataset

    @staticmethod
    def _get_monai_subject_dataset_by_match_dic(match_dic, transform):
        """
        :param match_dic: {subj_name: [img1_path, img2_path... mask_path]}
        :return:
        """
        # check whether a mask exists
        mask_path_test = match_dic[list(match_dic.keys())[0]]['mask']

        # get MONAI SubjectsDataset, preprocessing
        # convert dic to list
        subj_info_list = list(match_dic.values())
        img_modality_num = len(subj_info_list[0]["img"])
        # will be ["img", "img_1", "img_2", ...]
        for i in subj_info_list:
            for m_i in range(img_modality_num):
                i[f"img_{m_i}"] = i["img"][m_i]
            i["img"] = i["img_0"]
        
        if len(mask_path_test):
            for i in subj_info_list:
                i["mask"] = i["mask"][0]
        else:
            print('No mask input!')

        # get MONAI SubjectsDataset
        dataset = monai.data.Dataset(subj_info_list, transform=transform)
        print("MONAI dataset ready")

        return dataset

    def _get_custom_monai_transform_by_dic(self, trans_dic):
        trans_list = []
        for trans_name in trans_dic:
            transforms_i = getattr(monai.transforms, trans_name)
            trans_list.append(transforms_i(**trans_dic[trans_name]))

        self._set_logging('info', f"transform: {trans_list}")

        return monai.transforms.Compose(trans_list)
