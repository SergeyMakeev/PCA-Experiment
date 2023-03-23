import cv2
import numpy as np
from sklearn.decomposition import PCA

# packages to install
# numpy
# scikit-learn
# opencv-python


def test():
    image_raw = cv2.imread("red-bricks2_albedo.png")
    # print(image_raw.shape)
    # print(image_raw)

    # input: num_pixels x 3 (RGB)
    img = image_raw.reshape(-1, 3)
    # print(img.shape)
    # print(img)

    # normalize 0..1
    img = img / 255

    print("PCA")
    pca = PCA(n_components=2)
    # pca.fit(img)
    img_reduced = pca.fit_transform(img)
    # print(img_reduced.shape)
    # pad with zeroes, reshape, and save reduced image (for test)
    tmp_image = img_reduced * 255
    # print(tmp_image.shape)

    # add column (zero filled)
    tmp_image = np.hstack((tmp_image, np.atleast_2d(np.zeros(tmp_image.shape[0])).T))
    # tmp_image = np.hstack((tmp_image, np.atleast_2d(np.zeros(tmp_image.shape[0])).T))

    # print(tmp_image.shape)
    cv2.imwrite("tmp.png", tmp_image.reshape(image_raw.shape))

    print(pca.explained_variance_ratio_)
    print(pca.components_)

    img_reconstructed = pca.inverse_transform(img_reduced)
    # print(img_reconstructed.shape)

    # reshape back to 2d image
    img_r = img_reconstructed.reshape(image_raw.shape)
    img_r = img_r * 255
    # print(img_r.shape)

    cv2.imwrite("result.png", img_r)


test()
