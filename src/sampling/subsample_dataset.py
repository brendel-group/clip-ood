import io 
import os
import tarfile
import numpy as np
from argparse import ArgumentParser
import time


def are_files_selected(current_file_root, img_list):
    """This is the function that decides which pairs of images+metadata
    are written in the resulting tar and which are not.""" 

    # current_files.keys() contains e.g.
    #  dict_keys(['000000007.jpg', '000000007.json', '000000007.txt'])
    #  i.e. a 9-digit image ID and a file ending
    # the img list is a whitelist for all desired image IDs
    return int(current_file_root) in img_list


def write_files(current_files, out_tar_stream):
    """Appends the files to the tar archive.

    Args:
        current_files (dict): keys are filnames and values is the data (bytearray)
        out_tar_stream (IOStream): the archive where we *write* the files
    """
    for filename in current_files:
        data = current_files[filename]
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        out_tar_stream.addfile(info, io.BytesIO(data))
    

def process(archive_filename, out_tar_stream, img_list, **kwargs):
    """Iterates through the files inside the tar, groups them together based on prefix,
    filters the file-groups and finally writes them to the output tar.

    Args:
        archive_filename (str): the archive from which we *read* the files
        out_tar_stream (IOStream): the archive where we *write* the files
    """

    label_list = kwargs.pop('label_list', None)
    img_set = kwargs.pop('img_set', None)

    current_file_root = None
    current_files = {}
    count_records = 0

    print(f"Processing {archive_filename}")
    stream = tarfile.open(archive_filename, mode="r|*")
    for tarinfo in stream:
        file_root = ".".join(tarinfo.name.split(".")[:-1])
        
        if current_file_root is None:
            current_file_root = file_root

        if file_root != current_file_root:
            if are_files_selected(current_file_root, img_set):
                #  write the files and add an additional file_root.cls file for the label
                #  the same image might occur multiple times with different labels, add
                #  an entry for each
                if label_list is not None:
                    list_idcs = (img_list == int(current_file_root)).nonzero()
                    filename = f'{current_file_root}.cls'
                    labels = label_list[list_idcs]
                    for label in labels:
                        current_files[filename] = str(label).encode()
                        write_files(current_files, out_tar_stream)
                        count_records += 1
                else:
                    write_files(current_files, out_tar_stream)
                    count_records += 1

            current_file_root = file_root
            current_files = {}

        fname = tarinfo.name
        if not tarinfo.isreg():
            continue
        if fname is None:
            continue
        if fname.startswith("__") and fname.endswith("__"):  # meta files
            continue
        data = stream.extractfile(tarinfo).read()
        current_files[tarinfo.name] = data
        
        stream.members = []
                
    # do 1 more time for last group of files
    if len(current_files) > 0 and are_files_selected(current_file_root, img_set):
        if label_list is not None:
            list_idcs = (img_list == int(current_file_root)).nonzero()
            filename = f'{current_file_root}.cls'
            labels = label_list[list_idcs]
            for label in labels:
                current_files[filename] = str(label).encode()
                write_files(current_files, out_tar_stream)
                count_records += 1
        else:
            write_files(current_files, out_tar_stream)
            count_records += 1

    return count_records


def create_output_tar_stream(output_folder, worker_index, out_tar_index):
    os.makedirs(output_folder, exist_ok=True)
    output_tar = f"{output_folder}/{worker_index}_{out_tar_index}.tar"
    out_tar_stream = tarfile.TarFile(output_tar, 'w')
    return out_tar_stream


def main(args):
    # load image and label info

    # keep only the first n records
    #  the lists are already constrained to some size n, but for the
    #  per-query or per-class case there might be a few more entries
    #  than we need since the arrays are rectangular

    img_list = np.load(args.img_file, allow_pickle=True)
    img_list = img_list.ravel()[:args.n_imgs]
    img_set = set(img_list)  # convert it into set to make things faster
    if args.label_file is not None:
        label_list = np.load(args.label_file, allow_pickle=True)
        label_list = label_list.ravel()[:args.n_imgs]
        #label_list = set(label_list)
    else:
        label_list = None


    out_tar_index = 0
    out_tar_stream = create_output_tar_stream(
        args.out_dir, args.worker_idx, out_tar_index)
    count_records = 0

    tars = []
    for f in os.listdir(args.data_dir):
        if f.endswith(".tar"):
            tars.append(f)
    tars = sorted(tars)

    # Pay attention to this line. We are running it distributed 
    #   and the number of workers is "num_workers".
    # The current worker only picks the file at index=worker_index,
    #   and then skips "num_workers" files 
    #   (a cycle of files which are picked up by the other workers)
    for i in range(0+args.worker_idx, len(tars), args.n_workers):
        if label_list is not None:
            count_records += process(args.data_dir+"/"+tars[i], out_tar_stream, img_list, img_set=img_set,
                                     label_list=label_list)
        else:
            count_records += process(args.data_dir + "/" + tars[i], out_tar_stream, img_list, img_set=img_set)

        # Aim for the output tars to be ~2.5GB
        if count_records > args.n_per_tar:
            print(f"Wrote {count_records} records. Closing the tar and opening a new one.")
            out_tar_stream.close()
            out_tar_index += 1
            out_tar_stream = create_output_tar_stream(
                args.out_dir, args.worker_idx, out_tar_index)
            count_records = 0

    out_tar_stream.close()
    print(f"Job was successful. It wrote {out_tar_index+1} output tar files")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('worker_idx', type=int,
        help='index of the worker running this process')
    parser.add_argument('n_workers', type=int,
        help='number of workers in total')
    parser.add_argument('-d', '--data_dir', type=str,
        help='dataset to subsample')
    parser.add_argument('-o', '--out_dir',
        help='output folder')
    parser.add_argument('-i', '--img_file',
        help='numpy file with image paths to select from the dataset')
    parser.add_argument('-l', '--label_file', default=None,
        help='Give None if no numpy. Otherwise give numpy file with labels for each image')
    parser.add_argument('-n', '--n_imgs', type=int, default=15000000,
        help='number of images to sample')
    parser.add_argument('-n_per_tar', type=int, default=100000,
        help='number of images per tar ball')

    args = parser.parse_args()
    start_time = time.time()
    main(args)
    print(f"{args.n_imgs} datapoints are saved in {time.time()-start_time} seconds")
