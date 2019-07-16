#!/usr/bin/python
# -*- coding:utf-8 -*-
# @Time   : 2018/3/31 0031 17:51
# @Author : scw
# @File   : loadimageandtest.py
# 用于进行训练集图片的加载以及训练数据模型的编写

import os
import numpy as np
import tensorflow as tf
from shenjingwangluomodel import inference, N_CLASSES, losses, trainning, learning_rate, evaluation, MAX_STEP

# 设置批次加载图片的纬度
BATCH_SIZE = 5
# 生成批次队列中的容量缓存的大小
CAPACITY = 256
# 设置我自己要对图片进行统一大小的高和宽
IMG_W = 208
IMG_H = 208
# 存放用来训练的图片的路径
train_dir = 'E:/tensorflowdata/calldata/'

def get_files(file_dir):
    # 定义存放各类别数据和对应标签的列表，列表名对应你所需要分类的列别名
    # A5，A6等是我的数据集中要分类图片的名字
    CallPhone = []
    label_Call = []
    OthorPicture = []
    label_Other = []

    for file in os.listdir(file_dir):
        # 根据图片的名称，对图片进行提取，这里用.来进行划分
        name = file.split(sep='.')
        if name[0] == 'call':
            CallPhone.append(file_dir + file)
            label_Call.append('0')
        else:
            OthorPicture.append(file_dir + file)
            label_Other.append('1')
    # 这里一定要注意，如果是多分类问题的话，一定要将分类的标签从0开始。
    # 这里是2类，标签为0，1，我之前以为这个标签应该是随便设置的，结果就出现了Target[0] out of range的错误。

    # 打印出提取图片的情况，检测是否正确提取(主要是查看读取的图片数目是否正确)
    print('There are %d CallPhone\nThere are %d OtherPicture\n' \
          %(len(CallPhone), len(OthorPicture)))

    # 用来水平合并数组（沿着水平方向将数组堆叠起来）
    image_list = np.hstack((CallPhone, OthorPicture))  #图片水平堆叠(其实就是把图片叠在一起)
    label_list = np.hstack((label_Call, label_Other))  #对应的标签水平堆叠

    # 复制数组的内容
    temp = np.array([image_list,label_list])
    # 对数组的内容进行转置操作，即行变列，列变行
    temp = temp.transpose()
    # 打乱数组中的顺序
    np.random.shuffle(temp)

    # 用list存储打乱顺序后的图片的次序
    image_list = list(temp[:,0])
    # 用list存储打乱顺序后的图片对应的标签内容
    label_list = list(temp[:,1])
    # 把标签原来的字符串类型转为int类型
    label_list = [int(i) for i in label_list]
    # 返回两个list
    return image_list, label_list

# 生成相同大小的批次（下面是自己读函数的参数的大体注释），
# 作用在于：比如训练的图片有20000张，那么一次性读入内存是不行的，所以就通过这个方法
# image, label: 要生成batch的图像和标签list
# image_W, image_H: 图片的宽高
# batch_size: 每个batch有多少张图片
# capacity: 队列容量
# return: 图像和标签的batch
def get_batch(image,label, image_W, image_H, batch_size, capacity):

    # 将python中的list类型转换成tf能够识别的格式(#tf.cast()用来做类型转换)
    image = tf.cast(image,tf.string)  # 这里是转图像为tensorflow能够识别的格式
    label = tf.cast(label,tf.int32)   # 这里是转标签为tensorflow能够识别的格式

    # 生成队列
    input_queue = tf.train.slice_input_producer([image,label])

    # 通过队列的方式进行读取数据，这样的方式比平常的读取tensorflow会快很多，因为是通过多线程进行的读写
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小，因为原始 图片中可能存在很小或者很大的图片
    # image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)，查了下，这个API也是进行裁剪图片的
    # 但是上面这个方法与下面的方法就有点区别，因为上面的方法碰到大的，是从图像中心向四周裁剪，如果图片超过规定尺寸，最后只会剩中间区域的一部分
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 标准化图片数据,对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    # 通过tensorflow中的线程进行生成相同大小批次的图片和标签
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = capacity)

    # 这个的话主要就是把多张图片按照一定大小批次的进行生成，比如我们想60张图片为一组进行训练，那么就是60，如果是想100，那就为100
    label_batch = tf.reshape(label_batch,[batch_size])
    print(label_batch)
    # 获取两个batch，两个batch即为传入神经网络的数据，这个主要是用到后面卷积神经网络需要的参数
    # image_batch是一个4D的tensor，[batch, width, height, channels]，
    # label_batch是一个1D的tensor，[batch]
    return image_batch,label_batch


# 进行图片和标签的训练处理
def run_training():
    # 训练集的图片位置
    train_dir = 'E:/tensorflowdata/calldata/'
    # 训练结果的位置
    logs_train_dir = 'E:/tensorflowdata/traindata/'
    # 存放一些模型文件的目录
    train,train_label = get_files(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H,
                                                         BATCH_SIZE,
                                                         CAPACITY)
    # 得到softmax分类模型
    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
    # 计算模型的损失率
    train_loss = losses(train_logits,train_label_batch)
    # 通过AdamOptimizer优化器使损失率变小，所以这里就出现一个学习率的概念
    train_op = trainning(train_loss,learning_rate)
    # 得到训练效果
    train_acc = evaluation(train_logits,train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    # 初始化tensorflow中的参数的初始化，这里我用了全局变量进行初始化
    sess.run(tf.global_variables_initializer())
    # 初始化tensorflow的一个协调器，主要是为了后面的队列训练有用
    coord = tf.train.Coordinator()
    # 通过队列形式来进行训练，主要是提高训练的速度
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 下面主要是进行迭代的训练
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss, tra_acc = sess.run([train_op,train_loss,train_acc])
            # 每迭代50次，打印出一次结果
            if step % 50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))

                summary_str = sess.run(summary_op)
                # 把训练结果添加到训练队列，进行后续的继续迭代训练
                train_writer.add_summary(summary_str,step)
            # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
            if step % 200 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    # 判断是否有训练异常出现
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# 进行神经网络模型的训练，只有这样才能够获得到对输入数据的训练
# run_training()
