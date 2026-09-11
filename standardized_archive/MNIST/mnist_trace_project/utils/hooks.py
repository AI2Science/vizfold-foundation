def register_hooks(model):
    activations = {}

    def save_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    model.conv1.register_forward_hook(save_activation("conv1"))
    model.conv2.register_forward_hook(save_activation("conv2"))
    model.fc1.register_forward_hook(save_activation("fc1"))
    model.fc2.register_forward_hook(save_activation("fc2"))

    return activations