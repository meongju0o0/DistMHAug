import torch as th

def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation and test set.

    Parameters
    ----------
    model : DistSAGE
        The model to be evaluated.
    g : DistGraph
        The entire graph.
    inputs : DistTensor
        The feature data of all the nodes.
    labels : DistTensor
        The labels of all the nodes.
    val_nid : torch.Tensor
        The node IDs for validation.
    test_nid : torch.Tensor
        The node IDs for test.
    batch_size : int
        Batch size for evaluation.
    device : torch.Device
        The target device to evaluate on.

    Returns
    -------
    float
        Validation accuracy.
    float
        Test accuracy.
    """

    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid])

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    labels : torch.Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Accuracy.
    """

    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).to(dtype=th.float32).sum() / len(pred)
