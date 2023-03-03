'''
import mindspore as ms
import pickle
with open('a.pkl', 'rb') as file:
    new_ckpt = pickle.load(file)
ms.save_checkpoint(new_ckpt, 'ms_fs2.ckpt')

msf = open('ms.txt').readlines()
ptf = open('pt.txt').readlines()
ptc = torch.load('/home/zhudongyao/ptFastSpeech2/output/ckpt/LJSpeech/160000.pth.tar')['model']

new_ckpt = []
for msl, ptl in zip(msf, ptf):
    msl = msl.split(' ', maxsplit=1)[0]
    ptl = ptl.split(' ', maxsplit=1)[0]
    new_ckpt.append({'name': msl, 'data': ptc[ptl].detach().cpu().numpy()})
ms.save_checkpoint(new_ckpt, 'ms_fs2.ckpt')
'''

'''
import torch
msf = open('ms.txt').readlines()
ptf = open('pt.txt').readlines()
ptc = torch.load('/home/zhudongyao/ptFastSpeech2/output/ckpt/LJSpeech/160000.pth.tar')['model']

new_ckpt = []
for msl, ptl in zip(msf, ptf):
    msl = msl.split(' ', maxsplit=1)[0]
    ptl = ptl.split(' ', maxsplit=1)[0]
    new_ckpt.append({'name': msl, 'data': ptc[ptl].detach().cpu().numpy()})
import pickle
with open('a.pkl', 'wb') as file:
    pickle.dump(new_ckpt, file)
'''
import mindspore as ms
import pickle
with open('a.pkl', 'rb') as file:
    new_ckpt = pickle.load(file)
    for item in new_ckpt:
        t = ms.Tensor(item['data'])
        if 'position_enc' in item['name']:
            t = t.reshape([1001, 256])
        if len(t.shape) == 3:
            t = t.expand_dims(2)
        item['data'] = t
ms.save_checkpoint(new_ckpt, 'ms_fs2.ckpt')