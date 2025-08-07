import imageio
import numpy as np    
import os
import sys
import bz2
import argparse
import multiprocessing
import scipy.ndimage
import PIL.Image
import imageio
import cv2
import dlib

# Ensure the script is run from the correct directory
if not os.path.exists('src/utils'):
    print("Please run this script from the root directory of the project.")
    sys.exit(1)

# Argument parser for combining GIFs
parser = argparse.ArgumentParser(description='Combine two GIFs side by side', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--g1", required=True, help="First GIF file")
parser.add_argument("--g2", required=True, help="Second GIF file")
# Argument parser for aligning faces in images
# parser = argparse.ArgumentParser(description='Align faces from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--output_size', default=1024, help='The dimension of images for input to the model', type=int)
parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)
args, other_args = parser.parse_known_args()

# Function to combine two GIFs side by side
def combine_gifs(gif1_path, gif2_path, output_path='output.gif'): 
    """ Combines two GIFs side by side into a new GIF.
    Args:
        gif1_path (str): Path to the first GIF file.
        gif2_path (str): Path to the second GIF file.
        output_path (str): Path where the combined GIF will be saved.
    Returns:
        new_g (imageio.get_writer): Writer object for the new combined GIF."""
    g1 = imageio.get_reader(gif1_path)
    g2 = imageio.get_reader(gif2_path)
    
    num = min(g1.get_length(), g2.get_length()) 
    
    new_g = imageio.get_writer(output_path)
    
    for f in range(num):
        img1 = g1.get_next_data()
        img2 = g2.get_next_data()
        new_image = np.hstack((img1, img2))
        new_g.append_data(new_image)
    
    g1.close()
    g2.close()    
    new_g.close()
    
    return


# Code to unpack bz2 files 
def unpack_bz2(src_path):
    """ Unpacks a .bz2 compressed file.
    Args:
        src_path (str): Path to the .bz2 file.
        Returns:
        dst_path (str): Path to the unpacked file.
    """
        
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

# Align function from FFHQ dataset pre-processing step
def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
    """ Aligns a face in an image based on detected landmarks.
    Args:
        src_file (_type_): _description_
        dst_file (_type_): _description_
        face_landmarks (_type_): _description_
        output_size (int, optional): _description_. Defaults to 1024.
        transform_size (int, optional): _description_. Defaults to 4096.
        enable_padding (bool, optional): _description_. Defaults to True.
        x_scale (int, optional): _description_. Defaults to 1.
        y_scale (int, optional): _description_. Defaults to 1.
        em_scale (float, optional): _description_. Defaults to 0.1.
        alpha (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_"""
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file).convert('RGBA').convert('RGB')

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = PIL.Image.fromarray(img, 'RGBA')
        else:
            img = PIL.Image.fromarray(img, 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(dst_file, 'PNG')

class LandmarksDetector:
    def __init__(self, predictor_model_path='src/utils/shape_predictor_68_face_landmarks.dat'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")



# Align faces in images
def align_faces_in_images(RAW_IMAGES_DIR="data/images/", ALIGNED_IMAGES_DIR="data/aligned_images/", output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, use_alpha=args.use_alpha):
    """ Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step.
    Args:
        RAW_IMAGES_DIR (str): Directory with raw images for face alignment.
        ALIGNED_IMAGES_DIR (str): Directory for storing aligned images.
        output_size (int): The dimension of images for input to the model.
        x_scale (float): Scaling factor for x dimension.
        y_scale (float): Scaling factor for y dimension.
        em_scale (float): Scaling factor for eye-mouth distance.
        use_alpha (bool): Add an alpha channel for masking.
    Returns:
        None
    """
    # Initialize the landmarks detector
    landmarks_detector = LandmarksDetector()
    
    for img_name in os.listdir():
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            if os.path.isfile(fn):
                continue
            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    print('Starting face alignment...')
                    face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")
    return


#
def cross_dis(img1_url, img2_url):
    """" Combines two images using OpenCV's addWeighted function.
    Args:
        img1_url (str): URL of the first image.
        img2_url (str): URL of the second image.
        Returns:
        dst (numpy.ndarray): Combined image.
    """
    src1 = cv2.imread(img1_url)
    src2 = cv2.imread(img2_url)

    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return dst