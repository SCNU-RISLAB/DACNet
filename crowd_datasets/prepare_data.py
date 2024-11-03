from scipy.io import loadmat
import os
import argparse


def get_points(root_path, mat_path):
    # m = loadmat(os.path.join(root_path, mat_path))
    m = loadmat(mat_path)
    return m['image_info'][0][0][0][0][0]


def get_image_list(root_path, sub_path):
    images_path = os.path.join(root_path, sub_path, 'images')
    # images = [os.path.join(images_path, im) for im in os.listdir(os.path.join(root_path, images_path)) if 'jpg' in im]
    images = [os.path.join(images_path, im) for im in os.listdir(images_path) if 'jpg' in im]
    return images


def get_gt_from_image(image_path):
    gt_path = os.path.dirname(image_path.replace('images', 'ground_truth'))
    gt_filename = os.path.basename(image_path)
    gt_filename = 'GT_{}'.format(gt_filename.replace('jpg', 'mat'))
    return os.path.join(gt_path, gt_filename)


def export_dataset(root_path, part_name, output_path):
    if part_name not in ['A', 'B']:
        raise NotImplementedError('Supplied dataset part does not exist')

    dataset_splits = ['train', 'test']
    for split in dataset_splits:
        part_folder = 'part_{}_final'.format(part_name)
        sub_path = os.path.join(part_folder, '{}_data'.format(split))
        images = get_image_list(root_path, sub_path=sub_path)

        try:
            os.makedirs(os.path.join(output_path, sub_path))
        except FileExistsError:
            print('Warning, output path already exists, overwriting')

        list_file = []
        for image_path in images:
            gt_path = get_gt_from_image(image_path)
            gt = get_points(root_path, gt_path)

            # for each image, generate a txt file with annotations
            new_labels_file = os.path.join(output_path, sub_path,os.path.basename(image_path).replace('jpg', 'txt'))
            with open(new_labels_file, 'w') as fp:
                for p in gt:
                    fp.write('{} {}\n'.format(p[0], p[1]))
            # list_file.append((image_path, new_labels_file))
            im_sp=image_path.split('/')
            
            lb_sp=new_labels_file.split('/')
            list_file.append((os.path.join(im_sp[-3],im_sp[-2],im_sp[-1]), os.path.join(lb_sp[-2],lb_sp[-1])))

        # generate file with listing
        with open(os.path.join(output_path, part_folder,'{}.list'.format(split)), 'w') as fp:
            for item in list_file:
                fp.write('{} {}\n'.format(item[0], item[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default='crowd_datasets\ShanghaiTech_Crowd_Detecting', help="Root path for shanghai dataset")
    parser.add_argument("--part", default='B', help="which part to export (A or B)")
    parser.add_argument("--output_path", default='crowd_datasets\ShanghaiTech_Crowd_Detecting', help="Path to store results")
    args = parser.parse_args()
    export_dataset(args.root_path, args.part, args.output_path)