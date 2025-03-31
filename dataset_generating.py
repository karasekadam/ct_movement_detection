import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from image_dif import image_diff

def add_noise_to_image(image_path: str) -> None:
    if not image_path.endswith(".tiff") or image_path.endswith("_noise.tiff"):
        return
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    gauss_noise = np.zeros(image.shape, dtype=np.uint16)
    cv2.randn(gauss_noise, 0, 300)
    gn_img = cv2.add(image, gauss_noise)
    cv2.imwrite(image_path.replace(".tiff", "") + "_noise.tiff", gn_img)


def add_noisy_images(images_path: str) -> None:
    labels_df = pd.read_excel(images_path + "dataset_labels.xlsx")
    image_paths1 = labels_df['img1'].tolist()
    image_paths2 = labels_df['img2'].tolist()

    unique_images = list(set(image_paths1 + image_paths2))
    for image_path in unique_images:
        add_noise_to_image(images_path + "grayscale/" + image_path + ".tiff")

    initial_labels = labels_df.copy()
    for label_row in initial_labels.iterrows():
        df_to_append = pd.DataFrame([
            {
            "img1": label_row[1]["img1"] + "_noise",
            "img2": label_row[1]["img2"],
            "label": label_row[1]["label"]},
            {
            "img1": label_row[1]["img1"],
            "img2": label_row[1]["img2"] + "_noise",
            "label": label_row[1]["label"]},
            {
            "img1": label_row[1]["img1"] + "_noise",
            "img2": label_row[1]["img2"] + "_noise",
            "label": label_row[1]["label"]}
        ])
        labels_df = pd.concat([labels_df, df_to_append], ignore_index=True)

    for image in unique_images:
        df_to_append = pd.DataFrame([
            {
            "img1": image + "_noise",
            "img2": image,
            "label": 0},
            {
            "img1": image,
            "img2": image + "_noise",
            "label": 0},
            {
            "img1": image + "_noise",
            "img2": image + "_noise",
            "label": 0}
        ])
        labels_df = pd.concat([labels_df, df_to_append], ignore_index=True)

    labels_df.to_excel(images_path + "dataset_labels_noise.xlsx", index=False)

def convert_to_dif(images_dir_path: str, labels_path: str, results_dir_path: str) -> None:
    labels_df = pd.read_excel(labels_path)
    image_paths1 = labels_df['img1'].tolist()
    image_paths2 = labels_df['img2'].tolist()

    for i in range(len(image_paths1)):
        img1 = cv2.imread(images_dir_path + image_paths1[i] + ".tiff", cv2.IMREAD_ANYDEPTH)
        img2 = cv2.imread(images_dir_path + image_paths2[i] + ".tiff", cv2.IMREAD_ANYDEPTH)
        dif = image_diff(img1, img2)
        print(np.sum(dif))
        cv2.imwrite(results_dir_path + image_paths1[i] + "_dif_" + image_paths2[i] + ".tiff", dif)


# add_noisy_images("labeled_dataset/")
convert_to_dif("labeled_dataset/grayscale_test/", "labeled_dataset/test_dataset_labels.xlsx",
               "labeled_dataset/dif_test/")

