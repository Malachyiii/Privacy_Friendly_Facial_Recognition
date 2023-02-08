import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt


import torch
import timm
import torchvision
import torchvision.transforms as transforms

##Cuda checks
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

#args
class args:
    #Meta args
    num_workers = os.cpu_count()
    print_freq = 100
    
    #Data args
    image_size = 256
    batch_size = 4
    
    #Model Args
    model_name = "mobilevitv2_100"
    model_shape = np.nan
    pretrained = True
    num_classes = 0
    in_chans = 3
    
    #Training Args
    epochs = 5
    optimizer = 'torch.optim.AdamW'


#create the dataset
dataset = torchvision.datasets.ImageFolder(root = "./TrainingSet/",
                                           transform=transforms.Compose([
                                               transforms.Resize(args.image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=args.num_workers)

# Plot some training images
real_batch = next(iter(data_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


#create the model
model = timm.create_model(args.model_name, pretrained = args.pretrained, num_classes = args.num_classes, in_chans = args.in_chans)
model.cuda(device)

#get the current models shape
x = torch.randn(1,args.in_chans, args.image_size, args.image_size).cuda(device)
args.model_shape = model(x).shape[1]

############### Utility Functions ####################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # save the model state!
    torch.save(state, filename) 
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

############### Loss function block ##################
def loss_function(features, target, unique_people, people_vectors):
    losses = torch.empty((len(target)))
    
    for i in range(len(features)):
        
        if len(unique_people) == 0:
            unique_people.append(target[i])
            people_vectors = features[i].to(device).unsqueeze(0)
            losses[i] = torch.tensor([0])
            #print("First Person Added")
        else:
            distances = torch.cdist(features[i].unsqueeze(0), people_vectors)

            if torch.min(distances) < 5 and unique_people[torch.argmin(distances)] == target[i]:
                #print("Correct! (Positive)")
                losses[i] = torch.tensor([0])
            
            elif torch.min(distances) < 5 and target[i] not in unique_people:
                #print("False Positive")
                losses[i] = torch.mean(distances)
            
            elif torch.min(distances) < 5 and unique_people[torch.argmin(distances)] != target[i]:
                #print("Incorrect guess")
                loss = torch.cdist(features[i].unsqueeze(0), people_vectors[unique_people.index(target[i])].unsqueeze(0))
            
            elif torch.min(distances) > 5 and target[i] not in unique_people:
                #print("Correct! (Negative)")
                people_vectors = torch.cat((people_vectors,features[i].unsqueeze(0)), dim = 0)
                unique_people.append(target[i])
                #print(people_vectors)
                losses[i] = torch.tensor([0])
            
            elif torch.min(distances) > 5 and target[i] in unique_people:
                #print("False Negative")
                loss = torch.cdist(features[i].unsqueeze(0), people_vectors[unique_people.index(target[i])].unsqueeze(0))
                
                losses[i] = loss
            


                   
            #if (target in unique_people) and (target == unique_people[]
    return losses, unique_people, people_vectors
    
class MyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, target, unique_people, people_vectors):
        tensor, unique_people, people_vectors = loss_function(features, target, unique_people, people_vectors)
        ctx.save_for_backward(tensor)
        return tensor, unique_people, people_vectors

    @staticmethod
    def backward(ctx, grad_features, unique_people, people_vectors):
        result, = ctx.saved_tensors
        return grad_features * result, None, None, None
        


############### Training Block #######################
def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    
    ######################
    #Create my unique people list
    unique_people = []
    people_vectors = []

    ######################
    # switch model to train mode here
    model.train()
    ################

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #####################
        # send the images to cuda device
        if device is not None:
            images, target = images.cuda(device, non_blocking = True), target.cuda(device, non_blocking = True)
        # send the target to cuda device

        
        ####Utilizing PyTorch native AMP####
        
        # compute features
        features = model(images)
        # compute loss 
        

        loss, unique_people, people_vectors = MyLoss.apply(features, target, unique_people, people_vectors)


        # measure accuracy and record loss
        print(images.size()[0])
        print(loss)
        losses.update(torch.mean(loss), images.size()[0])
 
        #### zero out gradients in the optimier
        optimizer.zero_grad()
        
        ## backprop!
        loss.sum().backward()
        
        # update the weights!
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return

############### Validation Block #####################
def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')
    
    ######################
    #Create my unique people list
    unique_people = []
    people_vectors = []

    ######################
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            
            ### send the images and target to cuda
            images, target = images.to(device), target.to(device)

            # compute features
            features = model(images)

            # compute loss
            loss, unique_people, people_vectors = MyLoss.apply(features, target, unique_people, people_vectors)


            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(f' * val_loss {losses.avg:.3f}')

    return losses.avg

#Training loop
optimizer = eval(args.optimizer)(model.parameters())

best_loss = np.inf

for epoch in range(args.epochs):
    #adjust_learning_rate(optimizer, epoch)
    
    # train for one epoch
    train(data_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(data_loader, model)

    # remember best acc@1 and save checkpoint
    is_best = val_loss > best_loss
    best_loss = min(val_loss, best_loss)


    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.model_name,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
    }, is_best)
    