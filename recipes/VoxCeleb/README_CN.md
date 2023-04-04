# 说话人识别--使用VoxCeleb数据集.
这个目录包含了基于VoxCeleb数据集的说话人识别、验证的程序代码(http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)。

# 说话人验证--使用ECAPA-TDNN提取说话人特证
执行以下脚本来训练说话人特证 [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py`

对数据集voxceleb1、voceleb2来说，识别说话人的准确率在98%~99%。

训练说话人特征结束后, 通过余弦相似度来验证说话人, 预训练模型地址放在(https://download.mindspore.cn/toolkits/mindaudio/ecapatdnn/)。 可以下载预训练模型，执行以下脚本来验证说话人:

`python speaker_verification_cosine.py`

系统达成:
- EER = 1.50% (voxceleb1 + voxceleb2) with s-norm
- EER = 1.70% (voxceleb1 + voxceleb2) without s-norm

这些结果都是在官方的voxceleb1验证集上得到 (veri_test2.txt)。

我们使用5倍数据增强（需要2.6T磁盘空间）可以得到当前精度。如果想达到EER(0.8%), 需要50倍数据增强, 只需要把`ecapatdnn.yaml`文件中的超参数 `number_of_epochs` 修改为10即可。
当然, 50倍数据增强需要26T磁盘空间。

模型在Ascend910上 8卡训练 24小时得到。

# VoxCeleb1 + VoxCeleb2 准备
Voxceleb2 音频文件是m4a格式的，在送给MindAudio之前，所有文件必须先转换成wav文件格式。数据集准备请参考以下步骤:

1. 下载 Voxceleb1、Voxceleb2数据集
请参考右侧网站上指导下载数据集: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
主要是使用Voxceleb1中分离出来的部分音频数据来计算EER。


2. 将m4a文件转换成wav文件
Voxceleb2 是按m4a格式保存音频文件的，想要在MindAudio中使用它们，需要先把所有的m4a文件转换成wav文件。
此脚本调用ffmpeg来完成此转换(https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830)， 转换操作需要数小时，但只需要转换一次即可。


3. 将所有的wav文件放在wav文件夹下，目录结构 `voxceleb12/wav/id*/*.wav` (e.g, `voxceleb12/wav/id00012/21Uxsk56VDQ/00001.wav`)


4. 将下载的`voxceleb1/vox1_test_wav.zip` 压缩文件放到 voxceleb12目录


5. 解压 voxceleb1 test文件， 在voxceleb2 目录下运行 `unzip vox1_test_wav.zip`命令


6. 拷贝 `voxceleb1/vox1_dev_wav.zip` 文件到voxceleb12目录


7. 解压voxceleb1 dev文件, 在voxceleb12目录下执行 `unzip vox1_dev_wav.zip`命令


8. 同时解压 voxceleb1 dev文件、 test文件到目录 `voxceleb1/`. 解压后格式参考 `voxceleb1/wav/id*/*.wav`.


9. 数据增强需要使用 `rirs_noises.zip` 文件， 可以在此链接中下载:http://www.openslr.org/resources/28/rirs_noises.zip ，下载后把它放入 `voxceleb12/` 目录。

# 训练
Voxceleb1、Voxceleb2 数据集准备好之后, 可以直接运行下面脚本预处理音频数据、在单卡上训练说话人特证:

`python train_speaker_embeddings.py`

注意:Voxceleb1、Voxceleb2 数据集很大,预处理音频数据时间比较长，所以在此使用30个进程同时做音频预处理。

如果机器不支持, 请修改`ecapatdnn.yaml`文件中的`data_process_num`超参数，到机器支持为止。

如果预处理数据已经生成, 可以运行下面代码单独在单卡上训练:

`python train_speaker_embeddings.py --need_generate_data=False`

如果预处理数据已经生成，可以运行下面代码进行分布式多卡训练:

`bash ./run_distribute_train_ascend.sh hccl.json`

hccl.json 文件是使用 hccl 工具生成的,可以参考此文章实现 (https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。注意，此脚本中要使用hccl.json文件的绝对路径。

# 评估
模型训练结束之后, 可以运行下面脚本来做说话人验证:

`python speaker_verification_cosine.py`

当说话人验证数据预处理之后, 可以运行下面脚本跳过数据预处理，只做说话人验证:

`python speaker_verification_cosine.py --need_generate_data=False`
