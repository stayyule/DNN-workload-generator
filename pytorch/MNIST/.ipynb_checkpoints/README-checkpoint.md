# Workload emulation for NN data parallel training

![kccb](images/kccb.png)

When running distributed training jobs on multiple Graphics Processing Units (GPUs), synchronization of gradients
is crucial for efficient communication. In the AI world, one effective way to achieve this is by utilizing the
NCCL backend.

NCCL employs variant collective communication algorithms, including All-to-All and All-Reduce-Ring, among others.
These algorithms enable efficient data exchange between GPUs.

To achieve data parallelism with ring all-reduce using NCCL backend:

1.  **Data Splitting**: The entire dataset is split into smaller chunks or minibatches.
2.  **Model Replication**: A copy of the AI model is replicated on each machine or processing unit.
3.  **Ring All-Reduce**: Each machine processes its assigned minibatch and performs forward and backward passes
independently. The gradients are then aggregated using a ring all-reduce operation, which involves the following
steps:

- **Gradient Collection**: Each machine collects the gradients from its local minibatch.
- **Ring Communication**: Machines communicate with their neighbors in a ring topology to share the gradients.
- **All-Reduce**: The shared gradients are aggregated across the entire network using an all-reduce operation, which combines the gradients from each machine.
4.  **Model Update**: The aggregated gradients are used to update the AI model parameters.


## Trials


When loading data with a batch size, you'll obtain the batch count, which determines how many data samples are
processed in parallel during training. In other words, this represents the number of trials where data flows
through your model at once.

After each trial, collective communication events occur, facilitating synchronization and coordination among the
GPUs involved in the training process.

Below, you'll find code snippets that demonstrate how to obtain the batch count when loading data.
    
    # Download and load the training data
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8192, shuffle=True)
    
    # when running data parallel, collective happens for every batch
    batch_count = len(trainloader)
    print(f'batch count is {batch_count}')

The number of trials for emulation should correspond to the product of the batch count and epoch count.

    trails = batch_count * epochs

## Collective data size

The collective data size corresponds to the sizes of the weight and bias gradients. Before exploring this
further, let's examine how to obtain the tensor data size through code examples.

    def get_tensor_data_size(t):
        return t.nelement() * t.element_size()

When training your network, the collective data size can be calculated by summing the sizes of the weight and
bias gradients.

    data_size += get_tensor_data_size(model.fc1.weight.grad)
    data_size += get_tensor_data_size(model.fc1.bias.grad)

### Model definition

    from torch import nn, optim
    import torch.nn.functional as F
    
    # TODO: Define your network architecture here
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 10)
            
        def forward(self, x):
            # make sure input tensor is flattened
            x = x.view(x.shape[0], -1)
            
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.log_softmax(self.fc4(x), dim=1)
            
            return x

### Fetch data size
    
    # TODO: Create the network, define the criterion and optimizer
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # TODO: Train the network here, in case you don't have gpu with your env, use device = 'cpu'
    device = 'cuda'
        
    epochs = 1
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        data_size = []
        for images, labels in trainloader:
            images_device = images.to(device)
            labels_device = labels.to(device)
    
            log_ps = model(images_device)
            loss = criterion(log_ps, labels_device)
            
            optimizer.zero_grad()        
            loss.backward()
    
            data_size.append(get_tensor_data_size(model.fc1.weight.grad) + get_tensor_data_size(model.fc1.bias.grad))
            data_size.append(get_tensor_data_size(model.fc2.weight.grad) + get_tensor_data_size(model.fc2.bias.grad))
            data_size.append(get_tensor_data_size(model.fc3.weight.grad) + get_tensor_data_size(model.fc3.bias.grad))
            data_size.append(get_tensor_data_size(model.fc4.weight.grad) + get_tensor_data_size(model.fc4.bias.grad))
            break
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
    
    print(f'Data size of collective: {data_size}')

I have my model collective data size for `[fc1, fc2, fc3, fc4]`

        Data size of collective: [803840, 131584, 33024, 2600]

## Emulation

To evaluate the fabric environment before scaling up to multiple GPUs, 
you can emulate a collective workflow with KCCB. 
This allows you to assess the performance of your setup without incurring 
significant overheads.

When collective happens in backpropagation, you get last layer collective first

![data_parallel](images/data_parallel.png)

[original picture](astra-sim.github.io)


Open KCCB application, click `Data Size` and make a list with numbers of trials

    data_size = [fc4, fc3, fc2, fc1]
    trial = 8 (1 epoch)

![data size](/images/data_size.png)

The algorithm is modified to use `Ring all reduce` for data parallel 
and an emulated infrastructure is established with 4 GPUs, 
each located on a separate host, allowing for a distributed computing setup.
Then run trial.


### Result

![cdf](/images/cdf.png)


In many cases, there is a significant delay of approximately 20+ milliseconds
when performing collective operations within the tested fabric. However, it's not uncommon for this delay to be
accompanied by a longer tail of up to 4 or 12 seconds in some instances.


![delay](/images/delay.png)







