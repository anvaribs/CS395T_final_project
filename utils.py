#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import queue
import threading
import zipfile
import tqdm
import cv2
import numpy as np
import pickle
import cv2
import glob
import matplotlib.pyplot as plt
from pdb import set_trace


def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h-pad_bottom, pad_left:w-pad_right, :]


def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img


def apply_model_frzn(dataset_address, model, preprocess_for_model, extensions=(".jpg",), input_shape=(224, 224), batch_size=32):

    counter = 0
    num_exm = 10000
    img_embedding = np.ndarray((num_exm, 3*3*1024))
    img_ids = np.zeros(num_exm)
    img_filenames = []
    print("\n\n creating the feature vectors ...")
    for filename in glob.glob(dataset_address): 
        if 'val' in dataset_address:
            name = 'val'
        if 'train' in dataset_address:
            name = 'train'
        set_trace()
        img_id = int(filename.strip('.jpg').replace('{}/COCO_{}2014_'.format(dataset_address,name),''))
        set_trace()
        if counter == num_exm:
            break
        img = cv2.imread(filename)
        img = cv2.resize(img, (img_width, img_height))
        arr = np.array(img).reshape((img_width,img_height,3))
        arr = np.expand_dims(arr, axis=0)
        # normalizing the images
        arr = arr / 255
        
        # plt.imshow(img)
    #     intermediate_output = intermediate_layer_model.predict(arr)
        intermediate_output = model3.predict(arr)
        img_embedding[counter,:] = intermediate_output[0].flatten()
        img_ids[counter] = img_id
        img_filenames.append(filename)
        counter += 1
        
    print("\n\n saving the feature vectors ...")
    main_dir = './datasets/coco/extracted_2014/'
    np.save(main_dir+"img_embeddings_{}_2014".format(dataset), img_embedding)
    np.save(main_dir+"img_ids_{}_2014".format(dataset), img_ids)
    myFile = open(main_dir+'img_filenames_{}_2014.txt'.format(dataset), 'w')
    for item in img_filenames:
        myFile.write("%s\n" % item)






def apply_model(zip_fn, model, preprocess_for_model, extensions=(".jpg",), input_shape=(224, 224), batch_size=32):
    # queue for cropped images
    q = queue.Queue(maxsize=batch_size * 10)

    # when read thread put all images in queue
    read_thread_completed = threading.Event()

    # time for read thread to die
    kill_read_thread = threading.Event()

    def reading_thread(zip_fn):
        zf = zipfile.ZipFile(zip_fn)
        for fn in tqdm.tqdm_notebook(zf.namelist()):
            if kill_read_thread.is_set():
                break
            if os.path.splitext(fn)[-1] in extensions:
                buf = zf.read(fn)  # read raw bytes from zip for fn
                img = decode_image_from_buf(buf)  # decode raw bytes
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fn)[-1], img), timeout=1)  # put in queue
                    except queue.Full:
                        if kill_read_thread.is_set():
                            break
                        continue
                    break

        read_thread_completed.set()  # read all images

    # start reading thread
    t = threading.Thread(target=reading_thread, args=(zip_fn,))
    t.daemon = True
    t.start()

    img_fns = []
    img_embeddings = []

    batch_imgs = []

    def process_batch(batch_imgs):
        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_embeddings = model.predict(batch_imgs)
        img_embeddings.append(batch_embeddings)

    try:
        while True:
            try:
                fn, img = q.get(timeout=1)
            except queue.Empty:
                if read_thread_completed.is_set():
                    break
                continue
            img_fns.append(fn)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                process_batch(batch_imgs)
                batch_imgs = []
            q.task_done()
        # process last batch
        if len(batch_imgs):
            process_batch(batch_imgs)
    finally:
        kill_read_thread.set()
        t.join()

    q.join()

    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, img_fns


def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)
