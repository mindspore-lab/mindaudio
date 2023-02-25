# Ascend910 Performance Record

* Resource: Ascend 910; CPU 2.60GHz; 192cores; memory 755G; OS Euler2.8
* MindSpore version: [1.9.0](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/ascend/x86_64/mindspore_ascend-1.9.0-cp37-cp37m-linux_x86_64.whl)
* CANN version: [6.0.RC1.alpha005](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.0.RC1/Ascend-cann-toolkit_6.0.RC1_linux-x86_64.run)

## Conformer Result

* Feature info: using fbank feature, cmvn, online speed perturb
* Training info: lr 0.001, acc_grad 1, 240 epochs, 4 Ascend910
* Decoding info: ctc_weight 0.3, average_num 30
* Performance result: total_time 13h17min

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 5.05  |
| ctc prefix beam search    | 5.05  |
| attention decoder         | 5.00  |
| attention rescoring       | 4.73  |

## Transformer Result

* Feature info: using fbank feature, cmvn, online speed perturb
* Training info: lr 0.002, acc_grad 1, 240 epochs, 4 Ascend910
* Decoding info: ctc_weight 0.3, average_num 30
* Performance result: total_time 7h39min

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 6.11  |
| ctc prefix beam search    | 6.11  |
| attention decoder         | 5.50  |
| attention rescoring       | 5.39  |

# Ascend310P Performance Record

* Resource: Ascend310P; CPU 2.00GHz; 64cores; memory 250G; OS Euler2.8
* MindSpore_lite version: [1.9.0](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/lite/release/linux/x86_64/ascend/mindspore_lite-1.9.0-cp37-cp37m-linux_x86_64.whl)
* CANN version: [6.0.RC1.alpha005](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%206.0.RC1/Ascend-cann-toolkit_6.0.RC1_linux-x86_64.run)

## Conformer Result

* Feature info: using fbank feature, cmvn, online speed perturb
* Infer info: frame_bucket_limit 1200
* Performance result: Single batch 0.02s

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 5.05  |
| ctc prefix beam search    | 5.05  |
| attention decoder         | 5.00  |
| attention rescoring       | 4.73  |

## Transformer Result

* Feature info: using fbank feature, cmvn, online speed perturb
* Infer info: frame_bucket_limit 1200
* Performance result: single batch 0.006s

| decoding mode             | CER   |
|---------------------------|-------|
| ctc greedy search         | 6.11  |
| ctc prefix beam search    | 6.11  |
| attention decoder         | 5.50  |
| attention rescoring       | 5.39  |
