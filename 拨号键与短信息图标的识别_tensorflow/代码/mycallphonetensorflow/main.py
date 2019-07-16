#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/3/31 0031 17:50
# @Author : scw
# @File   : main.py
# 进行图片预测方法调用的文件
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from shenjingwangluomodel import inference
# 定义需要进行分类的种类，这里我是进行分两种，因为一种为是拨号键，另一种就是非拨号键
CallPhoneStyle = 2
# 进行测试的操作处理==========================
# 加载要进行测试的图片
def get_one_image(img_dir):
    image = Image.open(img_dir)
    # Image.open()
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    plt.imshow(image)
    image = image.resize([208, 208])
    image_arr = np.array(image)
    return image_arr

# 进行测试处理-------------------------------------------------------
def test(test_file):
    # 设置加载训练结果的文件目录(这个是需要之前就已经训练好的，别忘记)
    log_dir = 'E:/tensorflowdata/traindata/'
    # 打开要进行测试的图片
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        # 把要进行测试的图片转为tensorflow所支持的格式
        image = tf.cast(image_arr, tf.float32)
        # 将图片进行格式化的处理
        image = tf.image.per_image_standardization(image)
        # 将tensorflow的图片的格式参数，转变为shape格式的，好像就是去掉-1这样的列表
        image = tf.reshape(image, [1,208, 208, 3])
        # print(image.shape)

        # 参数CallPhoneStyle：表示的是分为两类
        p = inference(image, 1, CallPhoneStyle)  # 这是训练出一个神经网络的模型
        # 这里用到了softmax这个逻辑回归模型的处理
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 对tensorflow的训练参数进行初始化，使用默认的方式
            sess.run(tf.global_variables_initializer())
            # 判断是否有进行训练模型的设置，所以一定要之前就进行了模型的训练
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 调用saver.restore()函数，加载训练好的网络模型
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            print('预测的标签为：')
            if max_index == 0:
                print("是拨号键图片")
            else:
                print("是短信图片")
            # print(max_index)
            print('预测的分类结果每种的概率为：')
            print(prediction)
            # 我用0,1表示两种分类，这也是我在训练的时候就设置好的
            if max_index == 0:
                print('图片是拨号键图标的概率为 %.6f' %prediction[:, 0])
            elif max_index == 1:
                print('图片是短信它图标的概率为 %.6f' %prediction[:, 1])
# 进行图片预测
test('E:\\tensorflowdata\\testdata\\3.jpeg')


'''
# 测试自己的训练集的图片是不是已经加载成功（因为这个是进行训练的第一步）
train_dir = 'E:/tensorflowdata/calldata/'
BATCH_SIZE = 5
# 生成批次队列中的容量缓存的大小
CAPACITY = 256
# 设置我自己要对图片进行统一大小的高和宽
IMG_W = 208
IMG_H = 208
image_list,label_list = get_files(train_dir) # 加载训练集的图片和对应的标签
image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY) # 进行批次图片加载到内存中

# 这是打开一个session，主要是用于进行图片的显示效果的测试
with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 2:
            # 提取出两个batch的图片并可视化。
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''