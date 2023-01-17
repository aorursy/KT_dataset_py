import torch
def dice_channel_torch(probability, truth, threshold):

    batch_size = truth.shape[0]

    channel_num = truth.shape[1]

    mean_dice_channel = 0.

    with torch.no_grad():

        for i in range(batch_size):

            for j in range(channel_num):

                channel_dice = dice_single_channel(probability[i, j,:,:], truth[i, j, :, :], threshold)

                mean_dice_channel += channel_dice/(batch_size * channel_num)

    return mean_dice_channel





def dice_single_channel(probability, truth, threshold, eps = 1E-9):

    p = (probability.view(-1) > threshold).float()

    t = (truth.view(-1) > 0.5).float()

    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)

    return dice
batchsize = 16

channel_num = 4

probability = torch.rand(batchsize,channel_num,256,1600)

truth = torch.ones(batchsize,channel_num,256,1600)



dice = dice_channel_torch(probability, truth, 0.5)



print('Avg Dice score in this batch is {}'.format(dice))