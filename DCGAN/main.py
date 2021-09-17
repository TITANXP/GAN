import os
from glob import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DCGAN.dataset import make_anime_dataset
from DCGAN.model import Generator, Discriminator
import cv2


def save_result(val_out, val_block_size, image_path, color_mode):
    def process(image):
        image = ((image + 1.0) * 127.5).astype(np.uint8)
        return image

    preprocesed = process(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            single_row = np.array([])
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    cv2.imwrite(image_path, final_image)



def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    """
    判别器误差函数
    """
    #采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 生成图图片与0之间的误差
    d_loss_fake = celoss_zero(d_fake_logits)
    # 真实图片与1之间的误差
    d_loss_real = celoss_one(d_real_logits)
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时， 要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_one(d_fake_logits)
    return loss


def celoss_one(logits):
    """
    计算当前预测概率与标签1 之间的交叉熵损失
    """
    y = tf.ones_like(logits)
    loss = tf.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zero(logits):
    y = tf.zeros_like(logits)
    loss = tf.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    z_dim = 100 # 隐藏向量z的长度
    epochs = 3000000 # 训练步数
    batch_size = 64
    learning_rate = 0.0002
    is_training = True

    img_path = glob(r'D:\python_project\GAN\DCGAN\faces\*.jpg')
    print('images num:', len(img_path))

    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)

    sample = next(iter(dataset)) # 采样
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())
    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape = (4, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    # 分别为生成器和判别器创建优化器
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    generator.load_weights('tmp/generator.ckpt')
    discriminator.load_weights(('tmp/discriminator.ckpt'))

    g_losses, d_losses = [], []

    for epoch in range(epochs):
        # 1.训练判别器
        for _ in range(1):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            # 采样真实图片
            batch_x = next(db_iter)
            # 判别器前向计算
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        #2.训练生成器
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        # 采样真实图片
        batch_x = next(db_iter)
        # 生成器前向计算
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, ' d-loss:', float(d_loss), ' g-loss:', float(g_loss))
            # 可视化
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            save_img_path = os.path.join(r'D:\python_project\GAN\DCGAN','gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, save_img_path, color_mode='P')

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

        if epoch % 300 == 1:
            # print(d_losses)
            # print(g_losses)
            generator.save_weights('tmp/generator.ckpt')
            discriminator.save_weights('tmp/discriminator.ckpt')

