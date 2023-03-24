import cv2
import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import exposure
import matplotlib.pyplot as plt

# packages to install
#
# numpy
# scikit-learn
# scikit-image
# opencv-python


def convert_to_u8(img):
    return cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def posterize_img(img, num_colors):
    kmeans = KMeans(n_clusters=num_colors)
    labels = kmeans.fit_predict(img)
    quantized = kmeans.cluster_centers_.astype("uint8")[labels]
    return quantized


def quantize_to_2bit(img):
    quantized = img & 0b11000000
    return  quantized


def quantize_to_4bit(img):
    quantized = img & 0b11110000
    return quantized


# match using cumulative density function
# https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py#L6
def match_cdf(source, template):
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        # NOTE: this code path is not used!
        src_values, src_lookup, src_counts = np.unique(source.reshape(-1),
                                                       return_inverse=True,
                                                       return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1),
                                             return_counts=True)

        # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    # curve_coefficients = np.polyfit(tmpl_quantiles, tmpl_values, 8)
    # p = np.poly1d(curve_coefficients)
    #
    # _xp = np.linspace(0, 1, 255)
    # plt.plot(_xp, p(_xp), 'r+')

    # matched = p(src_lookup)

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    _mean1 = np.mean(source)
    _median1 = np.percentile(source, 50)
    _stddev1 = np.std(source)
    _min1 = np.min(source)
    _max1 = np.max(source)

    _mean2 = np.mean(template)
    _median2 = np.percentile(template, 50)
    _stddev2 = np.std(template)
    _min2 = np.min(template)
    _max2 = np.max(template)

    # plt.plot(src_quantiles, interp_a_values, '-x')
    # plt.plot(tmpl_quantiles, tmpl_values, 'r+')
    # plt.show()

    matched2 = interp_a_values[src_lookup].reshape(source.shape)
    return matched2


def remap(src, ref):
    return cv2.normalize(src=src, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def remap___(src, ref):
    src_x = np.delete(src, [1, 2], axis=2)
    src_y = np.delete(src, [0, 2], axis=2)
    src_z = np.delete(src, [0, 1], axis=2)

    ref_x = np.delete(ref, [1, 2], axis=2)
    ref_y = np.delete(ref, [0, 2], axis=2)
    ref_z = np.delete(ref, [0, 1], axis=2)

    matched_x = match_cdf(convert_to_u8(src_x), ref_x)
    matched_y = match_cdf(convert_to_u8(src_y), ref_y)
    matched_z = match_cdf(convert_to_u8(src_z), ref_z)

    # add back third dimension
    matched_x = np.reshape(matched_x, matched_x.shape + (1,))
    matched_y = np.reshape(matched_y, matched_y.shape + (1,))
    matched_z = np.reshape(matched_z, matched_z.shape + (1,))

#    matched_x.re

    img = np.append(np.append(matched_x, matched_y, 2), matched_z, 2)

    return cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def remap__(src, ref):
    src_x = np.delete(src, [1, 2], axis=2)
    src_y = np.delete(src, [0, 2], axis=2)
    src_z = np.delete(src, [0, 1], axis=2)

    ref_x = np.delete(ref, [1, 2], axis=2)
    ref_y = np.delete(ref, [0, 2], axis=2)
    ref_z = np.delete(ref, [0, 1], axis=2)

    matched_x = exposure.match_histograms(src_x, ref_x)
    matched_y = exposure.match_histograms(src_y, ref_y)
    matched_z = exposure.match_histograms(src_z, ref_z)

    img = np.append(np.append(matched_x, matched_y, 2), matched_z, 2)

    return cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def normalize_packed(img):
    r_min = np.min(img)
    r_max = np.max(img)
    r_range = r_max - r_min
    normalized_img = np.clip((img - r_min) / r_range, 0, 1) * 255
    return convert_to_u8(normalized_img), r_min, r_range


def normalize_img(img, weight):
    # convert to -1..1 and then multiply by weight
    normalized_img = (np.clip(img, 0, 255) / 127.5 - 1.0) * weight
    return normalized_img


def denormalize_img(img, weight):
    # convert to -1..1 and then multiply by weight
    denormalized_img = np.clip(img / weight, -1, 1) * 127.5 + 127.5
    return convert_to_u8(denormalized_img)


def test():
    print("Load images")

    #
    # https://stats.stackexchange.com/questions/175640/pca-loading-with-weights-on-samples
    #
    albedo_weight = 2.5
    normal_weight = 1
    metallic_weight = 1
    roughness_weight = 1

    # load textures as num_pixels x 3 (RGB) arrays
    albedo_orig = cv2.imread("02_BaseColor.png")
    albedo = albedo_orig.reshape(-1, 3)
    normal_orig = cv2.imread("02_Normal.png")
    normal = normal_orig.reshape(-1, 3)
    metallic_orig = cv2.imread("02_Metallic.png")
    metallic = metallic_orig.reshape(-1, 3)
    roughness_orig = cv2.imread("02_Roughness.png")
    roughness = roughness_orig.reshape(-1, 3)

    # metallic = quantize_to_2bit(metallic)
    metallic = posterize_img(metallic, 8)
    roughness = quantize_to_4bit(roughness)

    # cv2.imshow("Compressed (metallic q)", metallic.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3)))
    # cv2.imshow("Compressed (roughness q)", roughness.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3)))

    print("Trim unused/redundant channels")
    # remove unused channels
    # remove .B
    normal = np.delete(normal, 0, axis=1)
    # remove .GR
    metallic = np.delete(metallic, [1, 2], axis=1)
    # remove .GR
    roughness = np.delete(roughness, [1, 2], axis=1)

    print("Normalize")
    albedo = normalize_img(albedo, albedo_weight)
    normal = normalize_img(normal, normal_weight)
    metallic = normalize_img(metallic, metallic_weight)
    roughness = normalize_img(roughness, roughness_weight)

    print("Combine")
    # combine albedo + normal + metallic + roughness
    img = np.append(np.append(np.append(albedo, normal, 1), metallic, 1), roughness, 1)

    print("PCA")
    pca = PCA(n_components=6)
    # pca.fit(img)
    img_reduced = pca.fit_transform(img)
    # r_min = np.min(img_reduced)
    # r_max = np.max(img_reduced)
    # # print(r_min)
    # # print(r_max)
    # r_range = r_max - r_min
    # packed_image = np.clip((img_reduced - r_min) / r_range, 0, 1) * 255
    # packed_min2 = np.min(packed_image)
    # packed_max2 = np.max(packed_image)
    # # print(packed_min2)
    # # print(packed_max2)

    packed_image_rgb_a = np.delete(img_reduced, [3, 4, 5], axis=1)
    packed_image_rgb_a, min_a, range_a = normalize_packed(packed_image_rgb_a)
    packed_image_rgb_b = np.delete(img_reduced, [0, 1, 2], axis=1)
    packed_image_rgb_b, min_b, range_b = normalize_packed(packed_image_rgb_b)

    cv2.imwrite("_packed_rgb_a.png", packed_image_rgb_a.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3)))
    cv2.imwrite("_packed_rgb_b.png", packed_image_rgb_b.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3)))

    print(pca.explained_variance_ratio_)
    print(pca.components_.shape)
    print("PCA Components")
    print(pca.components_)
    print("PCA Mean")
    print(pca.mean_)

    print("Reconstruct original images")

    rgb_a = (packed_image_rgb_a / 255) * range_a + min_a
    rgb_b = (packed_image_rgb_b / 255) * range_b + min_b
    packed_image = np.append(rgb_a, rgb_b, 1)

    # packed_image = img_reduced
    img_reconstructed_2 = pca.inverse_transform(packed_image)
    # run inverse_transform manually (we would need to do that in a pixel shader)
    img_reconstructed = np.dot(packed_image, pca.components_) + pca.mean_

    # reshape back to 2d image
    albedo2 = np.delete(img_reconstructed, [3, 4, 5, 6], axis=1)
    albedo2 = denormalize_img(albedo2, albedo_weight)
    albedo2 = albedo2.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3))
    # cv2.imwrite("_albedo.png", albedo2)

    # convert to -1 .. 1
    normal_y = np.delete(img_reconstructed, [0, 1, 2, 4, 5, 6], axis=1)
    normal_y = denormalize_img(normal_y, normal_weight) / 127.5 - 1.0
    normal_x = np.delete(img_reconstructed, [0, 1, 2, 3, 5, 6], axis=1)
    normal_x = denormalize_img(normal_x, normal_weight) / 127.5 - 1.0
    # reconstruct Z
    normal_x2 = normal_x * normal_x
    normal_y2 = normal_y * normal_y
    normal_tmp = 1 - (normal_x2 + normal_y2)
    # to prevent sqrt(0)
    normal_tmp = np.clip(normal_tmp, 0.00000001, 99999999999999.0)
    normal_z = np.sqrt(normal_tmp)

    # merge normal components
    normal2_ = np.append(np.append(normal_z, normal_y, 1), normal_x, 1)
    normal2 = sklearn.preprocessing.normalize(normal2_, axis=1, norm='l1')
    normal2 = normal2 * 127.5 + 127.5
    normal2 = normal2.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3))
    # cv2.imwrite("_normal.png", normal2)

    metallic2 = np.delete(img_reconstructed, [0, 1, 2, 3, 4, 6], axis=1)
    metallic2 = denormalize_img(metallic2, metallic_weight)
    metallic2 = np.tile(metallic2, 3)
    metallic2 = metallic2.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3))
    # cv2.imwrite("_metallic.png", metallic2)

    roughness2 = np.delete(img_reconstructed, [0, 1, 2, 3, 4, 5], axis=1)
    roughness2 = denormalize_img(roughness2, roughness_weight)
    roughness2 = np.tile(roughness2, 3)
    roughness2 = roughness2.reshape((albedo_orig.shape[0], albedo_orig.shape[1], 3))
    # cv2.imwrite("_roughness.png", roughness2)

    # remap using histogram
    metallic3 = remap(metallic2, metallic_orig)
    albedo3 = remap(albedo2, albedo_orig)
    normal3 = remap(normal2, normal_orig)

    roughness3 = remap(roughness2, roughness_orig)

    cv2.imwrite("_albedo_.png", albedo3)
    cv2.imwrite("_normal_.png", normal3)
    cv2.imwrite("_metallic_.png", metallic3)
    cv2.imwrite("_roughness_.png", roughness3)

    # cv2.imshow("Source (albedo)", albedo_orig)
    # cv2.imshow("Compressed (albedo)", albedo3)
    # cv2.imshow("Source (normal)", normal_orig)
    # cv2.imshow("Compressed (normal)", normal3)
    # cv2.imshow("Source (roughness)", roughness_orig)
    # cv2.imshow("Compressed (roughness)", roughness3)
    # cv2.imshow("Source (metallic)", metallic_orig)
    # cv2.imshow("Compressed (metallic)", metallic3)
    #
    # cv2.waitKey(0)

    print("Done")


test()
