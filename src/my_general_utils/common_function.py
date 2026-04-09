import re
import os
from functools import reduce
import logging
import pickle


def match_source_and_mask(source_dir, mask_dir, pattern):
    # load and draw segment line in slice img
    source_dic = {re.findall(pattern, i.name)[0]: i.path for i in os.scandir(source_dir) if re.search(pattern, i.name)}
    mask_dic = {re.findall(pattern, i.name)[0]: i.path for i in os.scandir(mask_dir) if re.search(pattern, i.name)}

    # match
    match_pattern = set(source_dic.keys()) & set(mask_dic.keys())
    only_source = set(source_dic.keys()) - set(mask_dic.keys())
    only_mask = set(mask_dic.keys()) - set(source_dic.keys())

    print(f'Matched: {len(match_pattern)}\n\tonly source: {len(only_source)}\n\tonly mask: {len(only_mask)}')

    # match source and mask
    match_dic = {i: [source_dic[i], mask_dic[i]] for i in match_pattern}

    # print an example
    example_keys_list = list(match_dic.keys())
    example_keys_list.sort()

    if len(example_keys_list):  # if no match, example_keys_list[0] will return an error
        example_keys = example_keys_list[0]
        print('Example:\n\tkey:\n\t\t{}\n\titems:'.format(example_keys))
        for ex_i in match_dic[example_keys]:
            print('\t\t{}'.format(ex_i))

    return match_dic


def match_file_mult_dir(dir_list, pattern, key_pattern=None, verbose=True):
    """
    Save matched file path across mult directory in a dic (matched by specified pattern)
    It's a extension function of 'match_source_and_mask'
    :param dir_list: list, which contains many dir (str)
    :param pattern:  str or list or tuple. Subject name pattern, which be used to match files across all dir
                     If input str, can be only 1 group in pattern, and which group will be set tot the key of match_dic
                     If using list/tuple, the content must correspond to dir_list, and have same keywords.
                     If using mult-group in pattren, please using para 'key_group_num' and 'key_pattern'
    :param key_pattern: str, default is None. Only setting when pattern is list/tuple, and have mult-group.
                        This para is pattern of key in match_dic.
                        e.g. If input pattern is: [r"HEMA_(\d{5})_(\d)", r"HEMA-(\d{5})-(\d)"]
                        Can setting 'key_pattern' to 'HEMA={}={}'
                        Using {} in key_pattern to replace each keyword
                        At this time, supposing the file names are 'HEMA_00002_1' and 'HEMA-00002-1' respectively,
                        key of match_dic (which is output) is 'HEMA=00002=1'
    :param verbose: bool, default is True. is print match info

    :return: match_dic: dic, key is file pattern, contains are file paths
    """
    print('Match pattern: {} in {} dir'.format(pattern, len(dir_list)))
    # check is str or list
    if isinstance(pattern, str):
        pattern = [pattern] * len(dir_list)
    elif not (isinstance(pattern, list) or isinstance(pattern, tuple)):
        raise TypeError('Only input str, list, or tuple! Now input type: {}'.format(type(pattern)), pattern)

    # get match dic
    # check key_group_num and key_pattern
    dir_dic_list = []
    if isinstance(key_pattern, str):
        for n, dir_i in enumerate(dir_list):
            dir_dic_list.append({key_pattern.format(*re.findall(pattern[n], i.name)[0]): i.path
                                 for i in os.scandir(dir_i)
                                 if re.search(pattern[n], i.name)})
    else:
        # get dic of each dir
        for n, dir_i in enumerate(dir_list):
            dir_dic_list.append({re.findall(pattern[n], i.name)[0]: i.path for i in os.scandir(dir_i)
                                 if re.search(pattern[n], i.name)})

    # print files in each dir
    if verbose:
        print('Find files number:')
        for n, i in enumerate(dir_dic_list):
            print('\t{}: {}'.format(dir_list[n], len(i.keys())))

    # match
    dir_pattern_list = [set(i.keys()) for i in dir_dic_list]
    match_pattern = reduce(lambda x, y: x & y, dir_pattern_list)

    # match source and mask
    match_dic = {i: list(map(lambda x: x[i], dir_dic_list)) for i in match_pattern}

    # print match info and an example
    if verbose:
        # match info
        print('Matched: {}'.format(len(match_pattern)))
        for n, pat_set_i in enumerate(dir_pattern_list):
            print('\tUnmatched in {}: {}'.format(dir_list[n], len(dir_pattern_list[n] - match_pattern)))

        # an example
        example_keys_list = list(match_dic.keys())
        example_keys_list.sort()

        if len(example_keys_list):
            example_keys = example_keys_list[0]
            print('Example:\n\tkey:\n\t\t{}\n\titems:'.format(example_keys))
            for ex_i in match_dic[example_keys]:
                print('\t\t{}'.format(ex_i))

    return match_dic


def mkdir_all(dir_all, verbose=True):
    """
    mkdir by check, and can input dir list
    :param dir_all: can input str, list or tuple. (list and tuple must store the dir name)
    :param verbose: bool, is print mkdir info
    :return: No, only mkdir
    """
    # check is str or list
    if isinstance(dir_all, str):
        dir_all = [dir_all]
    elif not (isinstance(dir_all, list) or isinstance(dir_all, tuple)):
        raise TypeError('Only input str, list, and tuple! Now input is {}'.format(type(dir_all)), dir_all)

    # run mkdir
    for mkdir_i in dir_all:
        if not os.path.exists(mkdir_i):
            os.makedirs(mkdir_i)
            if verbose:
                print('mkdir: {}'.format(mkdir_i))

    return


def flatten_list(li):
    return sum(([x] if not isinstance(x, list) else flatten_list(x) for x in li), [])


def get_logger(logging_path, logger_name='', level=logging.INFO, is_console_out=True,
               file_fmt='%(asctime)s %(name)s %(levelname)s: %(message)s',
               console_fmt='%(asctime)s %(name)s %(levelname)s: %(message)s'):
    # set logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create handler for writing to log file
    file_handler = logging.FileHandler(filename=logging_path, mode='w')
    file_handler.setFormatter(logging.Formatter(file_fmt))
    logger.addHandler(file_handler)

    # out to console
    if is_console_out:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(console_fmt))
        logger.addHandler(console_handler)

    return logger


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_file = pickle.load(f)

    return pkl_file


def save_pkl(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

    return pkl_path


def inc_print(info, sep='-', length=20):
    print('{} {} {}'.format(sep*length, info, sep*length))


def assert_with_logging(logger, condition, message):
    if not condition:
        logger.error(message)
        raise AssertionError(message)


def log_or_print(logger, level, msg):
    if logger is not None:
        if level == 'debug':
            logger.debug(msg=msg)
        elif level == 'info':
            logger.info(msg=msg)
        elif level == 'warning':
            logger.warning(msg=msg)
        elif level == 'error':
            logger.error(msg=msg)
        elif level == 'critical':
            logger.critical(msg=msg)
            raise RuntimeError(msg)
        else:
            logger.error('ERROR setting LEVEL: {}!\n message: {}'.format(level, msg))
    else:
        print("{}: {}".format(level, msg))

    return msg

# Using "tuple(x)" will breaks up the string,
# e.g., tuple("xy") will return ("x", "y"); ("xy") return "xy"; (("xy"),) return (("xy"),)
# but this function, force_to_tuple("xy") will return ("xy", ),
# and force_to_tuple(("xy")) will also return ("xy")
def force_to_tuple(x):
    return x if isinstance(x, tuple) else (x, )

def pattern_extract(x: str, input_pattern: str, out_pattern: str):
    """
    Extracts the specified pattern from the input text and concatenates it.
    For example, 
        pattern_extract("6017863_20252_2_0", r"(\d{7})_\d{5}_(\d)_\d", "{}_{}")
    return is "6017863_2"

    Args:
        x (str): input string
        input_pattern (str): pattern of input string, must have tuple to get group
        out_pattern (str): output pattern, using {} to match each () in input_pattern

    Returns:
        str: The specified pattern of extraction from input string
    """
    return out_pattern.format(*force_to_tuple(re.findall(input_pattern, x)[0]))
