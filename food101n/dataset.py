import os
import numpy as np

def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def gen_train_list():
    img = 0
    root_data_path = 'your_fold/Food-101N_release/meta/verified_train.tsv'
    class_list_path = 'your_fold/Food-101N_release/meta/classes.txt'

    file_path_prefix = 'your_fold/Food-101N_release/images'

    map_name2cat = dict()
    with open(class_list_path) as fp:
        for i, line in enumerate(fp):
            row = line.strip()
            map_name2cat[row] = i
    num_class = len(map_name2cat)
    print('Num Classes: ', num_class)

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        fp.readline()  # skip first line

        for line in fp:
            row = line.split()
            class_name = row[0].split('/')[0]
            if os.path.join(file_path_prefix, row[0]) in all_image:
                img += 1
                targets.append(map_name2cat[class_name])
                img_list.append(os.path.join(file_path_prefix, row[0]))

    targets = np.array(targets)
    img_list = np.array(img_list)
    print('Num Train Images: ', len(img_list), 'img_exist: ', img, 'choice: ', img)

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'train_images'), img_list)
    np.save(os.path.join(save_dir, 'train_targets'), targets)

    return map_name2cat

def gen_test_list(arg_map_name2cat):
    map_name2cat = arg_map_name2cat
    root_data_path = 'your_fold/Food-101N_release/food-101/meta/test.txt'

    file_path_prefix = 'your_fold/Food-101N_release/food-101/images/'

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.strip().split('/')
            class_name = row[0]
            targets.append(map_name2cat[class_name])
            img_list.append(os.path.join(file_path_prefix, line.strip() + '.jpg'))

    targets = np.array(targets)
    img_list = np.array(img_list)

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'test_images'), img_list)
    np.save(os.path.join(save_dir, 'test_targets'), targets)

    print('Num Test Images: ', len(img_list))


def gen_meta_list(arg_map_name2cat):
    img = 0
    map_name2cat = arg_map_name2cat
    root_data_path = 'your_fold/Food-101N_release/meta/verified_val.tsv'

    file_path_prefix = 'your_fold/Food-101N_release/images/'

    targets = []
    img_list = []
    with open(root_data_path) as fp:
        for line in fp:
            row = line.split()
            if row[1] == '1':
                class_name = row[0].split('/')[0]
                path = row[0].split('/')[1]
                # print(label, path)
    #         class_name = row[0]
                if os.path.join(file_path_prefix, row[0]) in all_image:
                    img += 1
                targets.append(map_name2cat[class_name])
                img_list.append(os.path.join(file_path_prefix, row[0]))
    # print(targets, img_list)
    #
    targets = np.array(targets)
    img_list = np.array(img_list)

    save_dir = check_folder('./image_list')
    np.save(os.path.join(save_dir, 'meta_images'), img_list)
    np.save(os.path.join(save_dir, 'meta_targets'), targets)

    print('Num meta Images: ', len(img_list), 'img_exist:', img, 'choice:', img)


import glob
all_image = glob.glob('your_fold/Food-101N_release/images/*/*.jpg')
map_name2cat = gen_train_list()
gen_test_list(map_name2cat)
gen_meta_list(map_name2cat)