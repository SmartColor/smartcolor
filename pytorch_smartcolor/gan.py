from __future__ import absolute_import, division, print_function, unicode_literals
import time
import numpy as np
import pandas as pd
from xlrd import open_workbook
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from visualization import visualization
from generator import Generator
from discriminator import Discriminator

use_gpu = torch.cuda.is_available()
if (use_gpu):
    print('gpu is available')
wb = pd.read_excel("rgb.xlsx")
train_data = wb.values
train_data = (train_data - 127.5) / 127.5  # data normalization, from [0, 255] to [-1, +1]
# print(train_data)
train_data = train_data.reshape(train_data.shape[0], 1, 5, 3).astype('float32')  # ?????
train_data = torch.from_numpy(train_data)
if use_gpu:
    train_data = train_data.cuda()
BUFFER_SIZE = 5000  # 从源数据集取得样本数量
BATCH_SIZE = 256  # 每一批训练数据的数量：从BUFFER_SIZE中取，从buffer中取BATCH_SIZE个样本到batch中，buffer不足BUFFER_SIZE个样本，从源数据集提取
timestamp = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))


# （BUFFER_SIZE-BATCH_SIZE）个样本

# 反卷积
# 先将数据reshape成（16*5*3），之后再通过3次反卷积生成（5，3，1）的满足要求的数据大小。
# pytorch里数据shape为[batch_size,channels,H,W],而tf里默认[b,h,w,c]
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class ColorDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        x = self.X[idx]
        return x

    def __len__(self):
        return len(self.X)
    # 生成器


generator = Generator()
discriminator = Discriminator()
# generator = nn.Sequential(
#     nn.Linear(in_features=100, out_features=8 * 5 * 3, bias=False),
#     nn.BatchNorm1d(num_features=8 * 5 * 3),
#     nn.LeakyReLU(0.3),
#     Reshape(8, 5, 3),
#     nn.Dropout(0.5),
#     # nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1),
#     # nn.BatchNorm2d(num_features=8),
#     # nn.LeakyReLU(0.3),
#     nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1),
#     nn.BatchNorm2d(num_features=4),
#     nn.LeakyReLU(0.3),
#     nn.Dropout(0.5),
#     nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1), bias=False, padding=1),
#     nn.Tanh()
# )

# 测试generator
# generator.eval()
# noise = torch.randn(2,100)
# generated_data = generator(noise)
# print(generated_data)

# 判别器
# discriminator = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=(1, 1), padding=1),
#     nn.LeakyReLU(0.3),
#     # nn.Dropout(0.3),
#     # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=1),
#     # nn.LeakyReLU(0.3),
#     # nn.Dropout(0.3),
#     nn.Flatten(),
#     nn.Linear(in_features=5 * 3 * 4, out_features=1),
#     nn.Sigmoid()
# )
if use_gpu:
    generator = generator.cuda()
    discriminator = discriminator.cuda()

# 测试判别器
# discriminator.eval()
# decision = discriminator(generated_data)
# print(decision)

# loss function
cross_entropy = nn.BCELoss()


# 定义判别器损失函数。因为真实数据最后判别的结果必须要为1，因此real_loss的意思是真实的数据x和正确判别结果y（全部为1的矩阵)的loss值。而假的数据最后判别的结果是0，因此fake_loss的意思是生成的假数据x
# 和正确判别结果y（全部为0的矩阵的）的loss值。
def discriminator_loss(r_loss, f_loss):
    return r_loss + f_loss


def real_loss(real_output):
    # 真实数据接近于1 虚假数据接近于0  交叉熵越小 real_loss越小 说明他俩越接近
    # Use Soft and Noisy Labels
    noise = (torch.rand(real_output.shape[0], real_output.shape[1]) - 0.5) / 3
    if use_gpu:
        noise = noise.cuda()
    real_label_with_noise = torch.ones_like(real_output) + noise
    real_label_with_noise = real_label_with_noise.clamp(min=0.5, max=1)
    return cross_entropy(real_output, real_label_with_noise)


def fake_loss(fake_output):
    # real_output)生成一个数组，数组的大小和real_output一样 值全为1  它和real_output做交叉熵  如果real_loss越小 说明他俩越接近
    # Use Soft and Noisy Labels
    noise = (torch.rand(fake_output.shape[0], fake_output.shape[1])) / 3
    if use_gpu:
        noise = noise.cuda()

    fake_label_with_noise = torch.zeros_like(fake_output) + noise
    fake_label_with_noise = fake_label_with_noise.clamp(min=0, max=0.5)
    # print(noise_label_with_noise)

    return cross_entropy(fake_output, fake_label_with_noise)


# .定义生成器损失函数。因为要让生成器生成以假乱真的数据，如果判别器判别的结果为（1）真，则会传给生成器（1）真。因此在生成器中将判别器的判别结果和1（真）之间的loss即为我们的生成器损失函数。
def generator_loss(fake_output):
    noise = (torch.rand(fake_output.shape[0], fake_output.shape[1]) - 0.5) / 3
    if use_gpu:
        noise = noise.cuda()
    real_label_with_noise = torch.ones_like(fake_output) + noise

    real_label_with_noise = real_label_with_noise.clamp(min=0.5, max=1)
    return cross_entropy(fake_output, real_label_with_noise)


# 定义优化器
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=1e-3)

# 定义训练次数和噪声维度等参数
EPOCHS = 10  # 一共要训练的轮数
noise_dim = 100  # 噪声维度
num_examples_to_generate = 16  # 每次生成16组数据  ？

# 我们将重复使用该种子
seed = torch.randn(num_examples_to_generate, noise_dim)  # 将16*100的噪音扔到生成器里边

tb = SummaryWriter('runs/' + timestamp)
count = 0


def train_step(data):  # 将16*100的噪音扔到生成器里边生成一个数据  生成的数据给
    global count

    noise = torch.randn(BATCH_SIZE, noise_dim)
    if use_gpu:
        noise = noise.cuda()

    generator.train()
    discriminator.train()
    # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_data = generator(noise)  # 生成的数据

    real_output = discriminator(data)  # 真的数据的判别结果
    fake_output = discriminator(generated_data)  # 生成数据的判别结果

    gen_loss = generator_loss(fake_output)  # 生成器损失函数
    r_loss = real_loss(real_output)
    f_loss = fake_loss(fake_output)
    disc_loss = discriminator_loss(r_loss, f_loss)  # 判别器损失函数
    tb.add_scalar('Gen_Loss', gen_loss.item(), count)
    tb.add_scalar('Disc_real_Loss', r_loss.item(), count)
    tb.add_scalar('Disc_fake_Loss', f_loss.item(), count)

    gen_loss.backward(retain_graph=True)  # 生成器的梯度 根据损失去算梯度 根据梯度去优化函数
    disc_loss.backward(retain_graph=True)  # 判别器的梯度

    generator_optimizer.step()  # 对生成器进行优化
    discriminator_optimizer.step()  # 对判别器进行优化

    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    count = count + 1


# 定义模型训练函数。
def train(dataset, epochs):
    for epoch in range(epochs):  # 总训练的次数

        for b, rgb_batch in enumerate(dataset):  # 一次训练时训练所有的batch
            train_step(rgb_batch)

        # 每 15 个 epoch 保存一次模型  也就是每训练15次保存一次模型  每次输出多少组配色？
        # if (epoch + 1) % 15 == 0:
        #   checkpoint.save(file_prefix = checkpoint_prefix)
        # print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # 最后一个 epoch 结束后输出结果  用训练好的50次模型根据种子生成结果
    generate_and_save(generator, epochs, seed)


# 生成与保存
def generate_and_save(model, epoch, test_input):
    # 注意 training` 设定为 False
    # 因此，所有层都在推理模式下运行（batchnorm）。
    if use_gpu:
        test_input = test_input.cuda()
    # model.eval()
    with torch.no_grad():
        predictions = model(test_input)
    predictions = predictions.cpu().numpy() * 127.5 + 127.5  # 还原数据
    predictions = predictions.reshape(predictions.shape[0], 15, ).astype(np.int)
    df = pd.DataFrame(predictions)
    df.to_csv('output' + timestamp + '.csv', mode='a', header=False, index=None)


for it in range(1, 20):
    print(it)
    # 批量化和打乱数据
    data = ColorDataset(train_data)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    # 如果shuffle 的buffer_size=数据集样本数量，随机打乱整个数据集
    train(dataloader, EPOCHS)  # 进行模型训练，并查看最后一次的训练结果
tb.close()
visualization('output' + timestamp)
