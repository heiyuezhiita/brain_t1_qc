# 2025.03.02 custom sampler
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)
import numpy as np
from torch.utils.data import Sampler
import warnings

# Weighted sampling
# Minimize replacement (There are often duplicates of data with keeped labels), but will shuffle data
# Ensure that data with specified labels (keeped labels) are all used within an epoch
# NOTE: Set the label with a smaller number of samples as the label you want to protect
class WeightedKeepLabelSampler(Sampler):
    def __init__(self,
                 label_weights_dict: dict,
                 keep_label_list: list,
                 label_list: list,
                 num_samples: int,
                 generator=None,) -> None:
        """_summary_

        Args:
            label_weights_dict (dict): weight of each label. key is label, and value is weight
                                       e.g., {0: 1, 1: 0.5, 2: 1}
            keep_label_list (list): Cases with label in this list must be completely traversed in one epoch.
                                         Please only protect labels with relatively small sample sizes, otherwise maybe get error.
                                         e.g., [0, 1]
            label_list (list): labels of all cases (list)
            num_samples (int): number of samples to draw, default=`len(dataset)`.
            generator (Generator): Generator used in sampling.

        Raises:
            ValueError: _description_
        """
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
       
        self.label_weights_dict = {k: float(v) for k, v in label_weights_dict.items()}
        self.keep_label_list = keep_label_list
        self.label_list = np.array(label_list)
        self.num_samples = num_samples
        self.generator = generator
        
        # derived 
        self.weight_kp_label_total = np.sum([self.label_weights_dict[i] for i in self.keep_label_list])
        self.kp_scale_dic = self._get_kp_scale_dic()
        self.unkp_label_list = self._get_unkp_label_list()
        
    def __len__(self) -> int:
        return self.num_samples
    
    # custom iter, in fact, return are the index of cases
    def __iter__(self) -> Iterator[int]:
        # rand_tensor = torch.multinomial(
        #     self.weights, self.num_samples, self.replacement, generator=self.generator
        # )
        # yield from iter(rand_tensor.tolist())
        
        # base_label_ind is list
        base_label_ind = self._get_base_ind_block()
        base_len = len(base_label_ind)
        
        if base_len >= self.num_samples:
            #
            warnings.warn(
                f"len of base_label_ind ({base_len}) > self.num_samples ({self.num_samples}) \
                  This makes this sampler invalid and does not guarantee that the label will be maintained."
                )
            total_label_ind_list = base_label_ind[: self.num_samples]
        else:
            # Generate multiple basic blocks to compose the final output
            # Since a basic block has already been generated, there is no need +1 in "range"
            total_label_ind_list = base_label_ind
            for _ in range(self.num_samples // base_len):
                total_label_ind_list.extend(self._get_base_ind_block())
            
            # cut to fin lenght
            total_label_ind_list = total_label_ind_list[: self.num_samples]
        
        # fin output ind
        yield from iter(total_label_ind_list)
        
        
    def _get_unkp_label_list(self):
        # get all labels, and un-keeped label
        all_label_list = np.unique(self.label_list).tolist()
        unkp_label_list = list(set(all_label_list) - set(self.keep_label_list))
        
        return unkp_label_list

    def _get_kp_scale_dic(self):
        # get case number scale across keeped label
        raw_kp_scale_dic = {}
        for i in self.keep_label_list:
            kp_ind_i = np.flatnonzero(self.label_list == i)
            raw_scale_i = self.label_weights_dict[i] / len(kp_ind_i)
            raw_kp_scale_dic[i] = raw_scale_i
        # All cases need to be keeped, so scale is minimized to 1
        min_raw_kp_scale = min(raw_kp_scale_dic.values())
        kp_scale_dic = {k: v / min_raw_kp_scale for k, v in raw_kp_scale_dic.items()}
        
        return kp_scale_dic

    def _get_base_ind_block(self):
        # get base index of keeped label, and weighted
        weighted_kp_label_ind_list = []
        for i in self.keep_label_list:
            kp_ind_i = np.flatnonzero(self.label_list == i)
            # for keeped label, related weight must >= 1
            # Expand the length of inds to n (relative weight) times
            weighted_kp_ind_i = self._adjust_len_of_np_vec_by_scale(kp_ind_i, self.kp_scale_dic[i])
            weighted_kp_label_ind_list.append(weighted_kp_ind_i)
        base_weighted_kp_label_ind = np.concatenate(weighted_kp_label_ind_list)

        # get base scales and inds of cases with unkeeped labels
        weighted_unkp_label_ind_list = []
        for i in self.unkp_label_list:
            unkp_ind_i = np.flatnonzero(self.label_list == i)
            scale_i =  len(base_weighted_kp_label_ind) * self.label_weights_dict[i] / self.weight_kp_label_total / len(unkp_ind_i)
            weighted_unkp_ind_i = self._adjust_len_of_np_vec_by_scale(unkp_ind_i, scale_i)
            weighted_unkp_label_ind_list.append(weighted_unkp_ind_i)
        base_weighted_unkp_label_ind = np.concatenate(weighted_unkp_label_ind_list)
        
        # concate base label ind block, and shuffle
        base_label_ind = np.concatenate((base_weighted_kp_label_ind, base_weighted_unkp_label_ind))
        # You can comment out this line to test that the sampler output is as expected
        np.random.shuffle(base_label_ind)
        base_label_ind = base_label_ind.tolist()
        
        return base_label_ind

    @staticmethod
    # if n > 1, Duplicate the vector several times, with the fractional part randomly filled
    # if n < 1, will randomly selected
    def _adjust_len_of_np_vec_by_scale(vec, scale_n):
        length = len(vec)
        target_len = int(length * scale_n)
        if scale_n > 1:       
            # first, expand vec by repeat
            # At this time, the length is just right or not enough
            out_vec = np.tile(vec, target_len // length)
            
            # Then, random select to supplement remaining length
            if target_len > len(out_vec):
                diff_len = target_len - len(out_vec)
                supple_vec = np.random.choice(vec, diff_len, replace=False)
                out_vec = np.concatenate((out_vec, supple_vec))
        elif scale_n < 1:
            target_len = int(length * scale_n)
            out_vec = np.random.choice(vec, target_len, replace=False)
        else:
            out_vec = vec

        return out_vec
            
            

# # for test open ==============================================
# from torch.utils.data import DataLoader, Dataset
# labels = [0] * 10 + [1] * 30 + [2] * 5
# weights_dict = {0: 1, 1: 0.8, 2: 2}
# keep_labels = [0, 2]

# sampler = WeightedKeepLabelSampler(label_weights_dict=weights_dict, keep_label_list=keep_labels,
#                                    label_list=labels, num_samples=len(labels))
# dataset = list(zip(range(len(labels)), labels))  
# dataloader = DataLoader(dataset, batch_size=5, sampler=sampler)

# for epoch in range(3):
#     print(f"Epoch {epoch}")
#     for n, batch in enumerate(dataloader):
#         if n > 3: break
#         print(batch)
# # for test end ===============================================


