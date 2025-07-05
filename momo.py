import os
import shutil
import argparse
from tqdm import tqdm

def merge_datasets(input_dir, output_name):
    datasets = ['CSAW', 'DMID', 'DDSM']
    splits = ['train', 'val', 'test']

    # Tạo thư mục output trong cùng input_dir
    output_root = os.path.join(input_dir, output_name)
    os.makedirs(output_root, exist_ok=True)

    # 1️⃣ Tạo cấu trúc YOLO
    for split in splits:
        os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)

    # 2️⃣ Copy và thêm prefix tránh trùng tên
    print("📂 Copying files...")
    for dataset_idx, dataset in enumerate(datasets):
        dataset_path = os.path.join(input_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"⚠️ Bỏ qua {dataset}, không tồn tại trong {input_dir}.")
            continue
        for split_idx, split in enumerate(splits):
            img_dir = os.path.join(dataset_path, split, 'images')
            lbl_dir = os.path.join(dataset_path, split, 'labels')
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                print(f"⚠️ Bỏ qua {dataset}/{split}, thiếu images hoặc labels.")
                continue

            img_files = os.listdir(img_dir)
            lbl_files = os.listdir(lbl_dir)
            
            # Only show tqdm for the first split of each dataset
            if split_idx == 0:
                print(f"Copying {dataset}/{split} images...")
                for fname in tqdm(img_files, desc=f"{dataset}/{split} images", ncols=80):
                    src_img = os.path.join(img_dir, fname)
                    new_fname = f"{dataset}_{fname}"
                    dst_img = os.path.join(output_root, split, 'images', new_fname)
                    shutil.copy2(src_img, dst_img)
                print(f"Copying {dataset}/{split} labels...")
                for fname in tqdm(lbl_files, desc=f"{dataset}/{split} labels", ncols=80):
                    src_lbl = os.path.join(lbl_dir, fname)
                    new_fname = f"{dataset}_{fname}"
                    dst_lbl = os.path.join(output_root, split, 'labels', new_fname)
                    shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"Copying {dataset}/{split} images...")
                for fname in img_files:
                    src_img = os.path.join(img_dir, fname)
                    new_fname = f"{dataset}_{fname}"
                    dst_img = os.path.join(output_root, split, 'images', new_fname)
                    shutil.copy2(src_img, dst_img)
                print(f"Copying {dataset}/{split} labels...")
                for fname in lbl_files:
                    src_lbl = os.path.join(lbl_dir, fname)
                    new_fname = f"{dataset}_{fname}"
                    dst_lbl = os.path.join(output_root, split, 'labels', new_fname)
                    shutil.copy2(src_lbl, dst_lbl)

    # 3️⃣ Gộp .txt
    print("📝 Merging text files...")
    for split in splits:
        merged_txt = os.path.join(output_root, f"{split}.txt")
        with open(merged_txt, 'w') as outfile:
            for dataset in datasets:
                dataset_path = os.path.join(input_dir, dataset)
                txt_file = os.path.join(dataset_path, f"{split}.txt")
                if os.path.exists(txt_file):
                    with open(txt_file, 'r') as infile:
                        for line in infile:
                            line = line.strip()
                            if line == '':
                                continue
                            filename = os.path.basename(line)
                            new_filename = f"{dataset}_{filename}"
                            new_path = f"{output_root}/{split}/images/{new_filename}"
                            outfile.write(new_path + '\n')

    print(f"✅ Merge completed! Merged dataset saved in '{output_root}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSAW, DMID, DDSM YOLO datasets into a named folder in the same parent directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the parent directory containing CSAW, DMID, DDSM folders."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the new folder to store merged dataset."
    )
    args = parser.parse_args()
    merge_datasets(args.input_dir, args.name)