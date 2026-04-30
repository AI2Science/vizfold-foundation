def write_batch(root, start_idx, images, logits, preds, activations):
    batch_size = images.shape[0]
    end_idx = start_idx + batch_size

    root["inputs/images"][start_idx:end_idx] = images
    root["outputs/logits"][start_idx:end_idx] = logits
    root["outputs/predictions"][start_idx:end_idx] = preds
    root["activations/conv1"][start_idx:end_idx] = activations["conv1"]

    return end_idx