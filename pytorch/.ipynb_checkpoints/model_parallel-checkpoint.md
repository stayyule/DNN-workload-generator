# Model Parallelism

![model_parallel](images/model_parallel.png)

[picture from](astra-sim.github.io)

Model parallelism is a technique used to train or infer large neural models that cannot fit into a single GPU.
Instead of replicating the entire model on each GPU (as done in data parallelism), model parallelism splits the model across different GPUs.
Each GPU hosts a portion of the model, and intermediate outputs are moved across GPUs during the forward pass.
By distributing the model across multiple GPUs, we can handle larger models effectively.

**Two Paradigms of Model Parallelism**:
Model parallelism approaches can be broadly classified into two paradigms:

**Intra-operator parallelism**: This approach partitions individual layers or computational operators of the model. Communication is required to transform tensors between different distributed layouts within a device mesh.

**Inter-operator parallelism**: Here, the computational graph itself is partitioned. Communication is needed to exchange full tensors between pairs of devices.

Both intra-op and inter-op communication can be implemented using existing collective and point-to-point communication primitives.

We emulate the full tensors exchange in this workshop.


## Trials

When loading data with a batch size, you'll obtain the batch count, which determines how many data samples are
processed in parallel during training. In other words, this represents the number of trials where data flows
through your model at once.

After each trial, collective communication events occur, facilitating synchronization and coordination among the
GPUs involved in the training process.

Below, you'll find [code snippets](MNIST/data_parallel.ipynb) that demonstrate how to obtain the batch count when loading data.

The number of trials for emulation should correspond to the product of the batch count and epoch count.

    trails = batch_count * epochs


## Collective data size

The collective data size corresponds to the sizes of the output of every layer. Before exploring this
further, let's examine how to obtain the tensor data size through code examples.

    def get_tensor_data_size(t):
        return t.nelement() * t.element_size()

When training your network, the collective data size can be calculated by summing the sizes of the weight and
bias gradients.

    data_size += get_tensor_data_size(model.fc1.weight.grad)
    data_size += get_tensor_data_size(model.fc1.bias.grad)

See [Model definition and gradient data size](MNIST/data_parallel.ipynb) in notebook.

The data size for data parallel is the sum of all layer gradients data size, which is

    Data size of collective: 971048
