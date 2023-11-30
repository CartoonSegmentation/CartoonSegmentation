import albumentations as A
from albumentations import DualIAATransform, to_tuple
import imgaug.augmenters as iaa
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
import numpy as np

class IAAAffine2(DualIAATransform):
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.
    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualIAATransform):
    """Perform a random four point perspective transform of the input.
    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}
    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")


def get_bg_transforms(transform_variant, out_size):
    max_size = int(out_size * 1.2)
    if transform_variant == 'train':
        transform = [
            A.SmallestMaxSize(max_size, always_apply=True, interpolation=cv2.INTER_AREA),
            A.RandomResizedCrop(out_size, out_size, scale=(0.9, 1.5), p=1, ratio=(0.9, 1.1)),
        ]
    else:
        transform = [
            A.SmallestMaxSize(out_size, always_apply=True),
            A.RandomCrop(out_size, out_size, True),
        ]
    return A.Compose(transform)


def get_fg_transforms(out_size, scale_limit=(-0.85, -0.3), transform_variant='train'):
    if transform_variant == 'train':
        transform = [
            A.LongestMaxSize(out_size),
            A.RandomScale(scale_limit=scale_limit, always_apply=True, interpolation=cv2.INTER_AREA),
            IAAAffine2(scale=(1, 1),
                       rotate=(-15, 15),
                       shear=(-0.1, 0.1), p=0.3, mode='constant'),
            IAAPerspective2(scale=(0.0, 0.06), p=0.3, mode='constant'),
            A.HorizontalFlip(),
            A.ElasticTransform(alpha=0.3, sigma=15, alpha_affine=15, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.3)
        ]
    elif transform_variant == 'distort_only':
        transform = [
            IAAAffine2(scale=(1, 1),
                       shear=(-0.1, 0.1), p=0.3, mode='constant'),
            IAAPerspective2(scale=(0.0, 0.06), p=0.3, mode='constant'),
            A.HorizontalFlip(),
            A.ElasticTransform(alpha=0.3, sigma=15, alpha_affine=15, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, p=0.3)
        ]
    else:
        transform = [
            A.LongestMaxSize(out_size),
            A.RandomScale(scale_limit=scale_limit, always_apply=True, interpolation=cv2.INTER_LINEAR)
        ]
    return A.Compose(transform)


def get_transforms(transform_variant, out_size, to_float=True):
    if transform_variant == 'distortions':
        transform = [
            IAAAffine2(scale=(1, 1.3),
                       rotate=(-20, 20),
                       shear=(-0.1, 0.1), p=1, mode='constant'),
            IAAPerspective2(scale=(0.0, 0.06), p=0.3, mode='constant'),
            A.OpticalDistortion(),
            A.HorizontalFlip(),
            A.Sharpen(p=0.3),
            A.CLAHE(),
            A.GaussNoise(p=0.3),
            A.Posterize(),
            A.ElasticTransform(alpha=0.3, sigma=15, alpha_affine=15, border_mode=cv2.BORDER_CONSTANT),
        ]
    elif transform_variant == 'default':
        transform = [
            A.HorizontalFlip(),
            A.Rotate(20, p=0.3)
        ]
    elif transform_variant == 'identity':
        transform = [] 
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    if to_float:
        transform.append(A.ToFloat())
    return A.Compose(transform)


def get_template_transforms(transform_variant, out_size, to_float=True):
    if transform_variant == 'distortions':
        transform = [
            A.Cutout(p=0.3, max_w_size=30, max_h_size=30, num_holes=1),
            IAAAffine2(scale=(1, 1.3),
                       rotate=(-20, 20),
                       shear=(-0.1, 0.1), p=1, mode='constant'),
            IAAPerspective2(scale=(0.0, 0.06), p=0.3, mode='constant'),
            A.OpticalDistortion(),
            A.HorizontalFlip(),
            A.Sharpen(p=0.3),
            A.CLAHE(),
            A.GaussNoise(p=0.3),
            A.Posterize(),
            A.ElasticTransform(alpha=0.3, sigma=15, alpha_affine=15, border_mode=cv2.BORDER_CONSTANT),
        ]
    elif transform_variant == 'identity':
        transform = [] 
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    if to_float:
        transform.append(A.ToFloat())
    return A.Compose(transform)


def rotate_image(mat: np.ndarray, angle: float, alpha_crop: bool = False) -> np.ndarray:
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    
    if alpha_crop and len(rotated_mat.shape) == 3 and rotated_mat.shape[-1] == 4:
        x, y, w, h = cv2.boundingRect(rotated_mat[..., -1])
        rotated_mat = rotated_mat[y: y+h, x: x+w]
    
    return rotated_mat


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return (codebook[labels].reshape(w, h, -1) * 255).astype(np.uint8)

def quantize_image(image: np.ndarray, n_colors: int, method='kmeans', mask=None):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    image = np.array(image, dtype=np.float64) / 255

    if len(image.shape) == 3:
        w, h, d = tuple(image.shape)
    else:
        w, h = image.shape
        d = 1

    # assert d == 3
    image_array = image.reshape(-1, d)

    if method == 'kmeans':

        image_array_sample = None
        if mask is not None:
            ids  = np.where(mask)
            if len(ids[0]) > 10:
                bg = image[ids][::2]
                fg = image[np.where(mask == 0)]
                max_bg_num = int(fg.shape[0] * 1.5)
                if bg.shape[0] > max_bg_num:
                    bg = shuffle(bg, random_state=0, n_samples=max_bg_num)
                image_array_sample = np.concatenate((fg, bg), axis=0)
                if image_array_sample.shape[0] > 2048:
                    image_array_sample = shuffle(image_array_sample, random_state=0, n_samples=2048)
                else:
                    image_array_sample = None

        if image_array_sample is None:
            image_array_sample = shuffle(image_array, random_state=0, n_samples=2048)
        
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(
            image_array_sample
        )

        labels = kmeans.predict(image_array)
        quantized  = recreate_image(kmeans.cluster_centers_, labels, w, h)
        return quantized, kmeans.cluster_centers_, labels

    else:

        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

        return [recreate_image(codebook_random, labels_random, w, h)]


def resize2height(img: np.ndarray, height: int):
    im_h, im_w = img.shape[:2]
    if im_h > height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    if im_h != height:
        img = cv2.resize(img, (int(height / im_h * im_w), height), interpolation=interpolation)
    return img

if __name__ == '__main__':
    import os.path as osp

    img_path = r'tmp\megumin.png'
    save_dir = r'tmp'
    sample_num = 24

    tv = 'distortions'
    out_size = 224
    transforms = get_transforms(tv, out_size ,to_float=False)
    img = cv2.imread(img_path)
    for idx in tqdm(range(sample_num)):
        transformed = transforms(image=img)['image']
        print(transformed.shape)
        cv2.imwrite(osp.join(save_dir, str(idx)+'-transform.jpg'), transformed)
    # cv2.waitKey(0)
    pass