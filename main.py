import torch
from torch.utils.data import Dataset, DataLoader
import tensorboardX
from data import MyDataset
from model import lipreading
import torch.nn as nn
from torch import optim
import os
import time
from tensorboardX import SummaryWriter
import tensorflow as tf

if(__name__ == '__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('options')


def data_from_opt(txt_path, folds):
    dataset = MyDataset( 
        txt_path,
        opt.vid_path,
        opt.vid_pad,
        opt.txt_pad,
        folds)
    print('vid_path:{},num_data:{}'.format(txt_path,len(dataset.data)))
    
    loader = DataLoader(dataset, 
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True)   
    return (dataset, loader)


# def data_loader(vid_path):
#     dsets = {x: MyDataset(x, vid_path, opt.vid_pad, opt.txt_pad) for x in ['train', 'val', 'test']}
#     dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], 
#                                         batch_size=16, 
#                                         shuffle=True, 
#                                         num_workers=4) for x in ['train', 'val', 'test']}
#     dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
#     print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
#     return dset_loaders, dset_sizes

# file1 = '/home/luomingshuang/s2s-lipreading-lrw/result_train1.txt'
# file2 = '/home/luomingshuang/s2s-lipreading-lrw/result_val1.txt'
# file3 = '/home/luomingshuang/s2s-lipreading-lrw/result_test1.txt'
# f1 = open(file1, 'a')
# f2 = open(file2, 'a')
# f3 = open(file3, 'a')

if(__name__ == '__main__'):
    model = lipreading(mode=opt.mode, nClasses=30).cuda()
    
    writer_1 = tf.summary.FileWriter("./logs5/plot_1")
    #writer_2 = tf.summary.FileWriter("./logs1/plot_2")
    log_var = tf.Variable(0.0)
    tf.summary.scalar("train_loss", log_var)
    writer_op = tf.summary.merge_all()

    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # writer = SummaryWriter('./logs')

    if(hasattr(opt, 'weights1')):
        pretrained_dict = torch.load(opt.weights1)
        model_dict = model.state_dict()
        # print(model_dict.keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)       

    (train_dataset, train_loader) = data_from_opt(opt.trn_txt_path, 'train')
    (val_dataset, val_loader) = data_from_opt(opt.val_txt_path, 'val')
    (tst_dataset, tst_loader) = data_from_opt(opt.tst_txt_path, 'test')
    
    criterion = nn.NLLLoss() 

    optimizer = optim.Adam(model.parameters(),
             lr=opt.lr,
             weight_decay=0)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    iteration = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()

        for (i, batch) in enumerate(train_loader):
            (encoder_tensor, decoder_tensor) = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
            # print('decoder_tensor size is:', decoder_tensor.size())
            # print('encoder_tensor:', encoder_tensor.shape)
            outputs = model(encoder_tensor, decoder_tensor, opt.teacher_forcing_ratio)            
            flatten_outputs = outputs.view(-1, outputs.size(2))
            # print('flatten_outputs:', flatten_outputs.shape, flatten_outputs)
            # print('decoder_tensor:', decoder_tensor.view(-1).shape, decoder_tensor.view(-1))
            loss1 = criterion(flatten_outputs, decoder_tensor.view(-1))
            # loss2 = rl_loss
            loss = loss1
            # print('loss1:', loss1, 'loss2:', loss2)
            # loss = loss2
            # loss = (1 - 1/(2000*(9.2))) * loss1 + (1 / (2000*(9.2))) * loss2
            optimizer.zero_grad()   

            # writer.add_scalars('/')
            summary = session.run(writer_op, {log_var: loss1.detach().cpu().numpy()})
            writer_1.add_summary(summary, iteration)
            writer_1.flush()

            # summary = session.run(writer_op, {log_var: loss2.detach().cpu().numpy()})
            # writer_2.add_summary(summary, iteration)
            # writer_2.flush()

            iteration += 1

            loss.backward()
            optimizer.step()
            tot_iter = epoch*len(train_loader)+i
            
            # if(i % opt.display == 0):
            #     speed = (time.time()-start_time)/(i+1)
            #     eta = speed*(len(train_loader)-i)
            #     print('tot_iter:{},loss:{},eta:{}'.format(tot_iter,loss,eta/3600.0))
            train_loss = loss.item()
            # txt1 = 'iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss)
            # f1.write('\n{}'.format(txt1))
            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

            # if(tot_iter % opt.test_iter == 0):
            # if (iteration % 20 == 0):
            if (iteration % 5000) == 0:
                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for (idx,batch) in enumerate(val_loader):
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
                        truth_txt = MyDataset.arr2txt(decoder_tensor)
                        predict_txt_total.extend(predict_txt)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
                    # print('{:<50}|{:>50}'.format(predict, truth))
                    print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                cer = MyDataset.cer(predict_txt_total, truth_txt_total)                
                print('cer:{}, wer:{}'.format(cer, wer))          
                print(''.join(101*'-'))
                # txt2 = 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer)
                # f2.write('\n{}'.format(txt2))
                savename = os.path.join(opt.save_dir1, 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)

        if epoch % 1 == 0:
                with torch.no_grad():
                    predict_txt_total = []
                    truth_txt_total = []
                    for (idx,batch) in enumerate(tst_loader):
                        (encoder_tensor, decoder_tensor) \
                            = batch['encoder_tensor'].cuda(), batch['decoder_tensor'].cuda()
                        outputs = model(encoder_tensor)
                        predict_txt = MyDataset.arr2txt(outputs.argmax(-1))
                        truth_txt = MyDataset.arr2txt(decoder_tensor)
                        predict_txt_total.extend(predict_txt)
                        truth_txt_total.extend(truth_txt)
                
                print(''.join(101*'-'))                
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-')) 
                
                for (predict, truth) in list(zip(predict_txt_total, truth_txt_total))[:10]:
                    # print('{:<50}|{:>50}'.format(predict, truth))
                    print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))                
                print(''.join(101 *'-'))
                wer = MyDataset.wer(predict_txt_total, truth_txt_total)
                cer = MyDataset.cer(predict_txt_total, truth_txt_total)                
                print('cer:{}, wer:{}'.format(cer, wer))          
                print(''.join(101*'-'))
                # txt3 = 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer)
                # f3.write('\n{}'.format(txt3))
                savename = os.path.join(opt.save_dir2, 'iteration_{}_epoch_{}_cer_{}_wer_{}.pt'.format(iteration, epoch, cer, wer))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename) 

# f1.close()
# f2.close()
# f3.close()
