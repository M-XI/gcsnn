import torch
import torchvision

def get_RotMNIST(dataset_path):
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                    ])

    # Random rotation
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.RandomRotation(
                                                        [0, 360],
                                                        torchvision.transforms.InterpolationMode.BILINEAR,
                                                        fill=0),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
    
    train_ds = torchvision.datasets.MNIST(
        root=dataset_path, 
        train=True, 
        transform=train_transform, 
        download=True
        )
    test_ds = torchvision.datasets.MNIST(
        root=dataset_path, 
        train=False, 
        transform=test_transform
        )
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_ds, val_ds = torch.utils.data.random_split(test_ds, [0.5, 0.5])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader