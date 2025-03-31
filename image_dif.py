import cv2
import numpy as np
import os


image_dir_1 = "sample stabilization test projections/fiducials/automatic acquisition/projections_Fibres_plasticfiller_16-02_15-13-32/"
image_dir_2 = "sample stabilization test projections/fiducials/automatic acquisition/projections_Fibres_plasticfiller_2_17-02_13-50-50/"
image_dir_3 = "sample stabilization test projections/fiducials/automatic acquisition/projections_nanovrstva_2_23-09_11-28-31/"
image_dir_4 = "sample stabilization test projections/fiducials/automatic acquisition/projections_nanovrstva_3_23-09_13-02-41/"
image_dir_5 = "sample stabilization test projections/fiducials/automatic acquisition/rubber/"
image_dir_6 = "sample stabilization test projections/fiducials/automatic acquisition/silicone_1/"
image_dir_7 = "sample stabilization test projections/fiducials/automatic acquisition/silicone_2/"
image_dir_8 = "sample stabilization test projections/fiducials/manual acquisition/cork/"
image_dir_9 = "sample stabilization test projections/no fiducials/automatic acquisition/cork_2/"
image_dir_10 = "sample stabilization test projections/fiducials/manual acquisition/cork/"
image_dir_11 = "sample stabilization test projections/fiducials/manual acquisition/fiducial_Case1/"
image_dir_12 = "sample stabilization test projections/fiducials/manual acquisition/seed/"
image_dir_13 = "sample stabilization test projections/fiducials/manual acquisition/fiducial_Case2/"


def equalize_image_16bit(image1, image2):
    # Compute the histogram and CDF of image2 (target histogram)
    hist2, bins2 = np.histogram(image2.flatten(), 65536, [0, 65536])
    cdf2 = hist2.cumsum()  # Cumulative distribution function
    cdf2_normalized = cdf2 * hist2.max() / cdf2.max()  # Normalize

    # Histogram of image1
    hist1, bins1 = np.histogram(image1.flatten(), 65536, [0, 65536])
    cdf1 = hist1.cumsum()  # CDF of image1

    # Normalize image1's intensity values based on image2's histogram
    cdf1_normalized = np.ma.masked_equal(cdf1, 0)  # Mask zeros to avoid division by zero
    cdf1_normalized = (cdf1_normalized - cdf1_normalized.min()) * 65535 / (
                cdf1_normalized.max() - cdf1_normalized.min())
    cdf1_final = np.ma.filled(cdf1_normalized, 0).astype(np.uint16)

    # Map the intensity values of image1 to match image2's histogram
    image1_normalized = cdf1_final[image1]
    return image1_normalized


def normalize_image_16bit(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    return normalized_image


def image_diff(img1: np.ndarray, img2: np.ndarray, show: bool=False) -> np.ndarray:
    img1_norm = (img1/256).astype('uint8')
    img2_norm = (img2/256).astype('uint8')
    # img1_norm = normalize_image_16bit(img1)
    # img2_norm = normalize_image_16bit(img2)
    # print(f"img1_mean: {np.mean(img1)}, img1_std: {np.std(img1)}")
    # print(f"img2_mean: {np.mean(img2)}, img2_std: {np.std(img2)}")
    # print(f"img1_norm_mean: {np.mean(img1_norm)}, img1_norm_std: {np.std(img1_norm)}")
    # print(f"img2_norm_mean: {np.mean(img2_norm)}, img2_norm_std: {np.std(img2_norm)}")
    difference = cv2.absdiff(img1_norm, img2_norm)

    # Threshold to highlight regions of change
    thresholded = None
    thresh = 15
    change_ratio = 0
    while (thresholded is None or change_ratio < 0.05) and thresh > 0:
        _, thresholded = cv2.threshold(difference, thresh, 256, cv2.THRESH_BINARY)
        thresh -= 1
        change_ratio = np.sum(thresholded) / (825 * 625)

    # Display or analyze the result
    if show:
        thresholded = cv2.resize(thresholded, (825, 625))
        print(np.sum(thresholded))
        print(np.sum(thresholded) / (825 * 625))
        print()
        # edge = canny_edge_detection(thresholded)
        # cv2.imshow('Edge', edge)
        cv2.imshow('Difference', thresholded)
        cv2.waitKey(0)

    return thresholded


def extract_keys(file):
    file = file.replace(".tiff", "")
    file = file.replace("_noise", "")

    int_part = 0
    string_part = file
    splitted = file.split("_")
    try:
        int_part = int(splitted[-1])
    except ValueError:
        int_part = 0
    string_part = "_".join(splitted[:-1])

    return string_part, int_part


def show_diff_of_image_dir(curr_dir):
    for filename in sorted(os.listdir(curr_dir), key=extract_keys):
        if filename.endswith(".tiff"):
            print(curr_dir + filename)
            img2 = cv2.imread(curr_dir + filename, cv2.IMREAD_ANYDEPTH)
            if 'img1' not in locals():
                img1 = img2
                continue
            image_diff(img1, img2, show=True)
            img1 = img2


def canny_edge_detection(img):
    img_blured = cv2.GaussianBlur(img, (3, 3), 0)
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold
    edge = cv2.Canny(img_blured, t_lower, t_upper)
    return edge


if __name__ == "__main__":
    show_diff_of_image_dir(image_dir_13)
