from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size, augmentation):
    train_steps = [
        transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if augmentation == "strong":
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
    else:
        train_steps.append(
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if augmentation == "strong":
        train_steps.append(
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), ratio=(0.5, 2.0))
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_transform
