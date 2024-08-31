    
import os
import random
import torch
import cv2 as cv
from PIL import Image
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Iterable, Dict
from classes import ImageDataset
from scipy.ndimage.filters import gaussian_filter1d

# IMPORTANT: CHANGE THESE
IMAGE_FILEPATH = "./imagenet/imagenet_images/"
LABEL_FILEPATH = "./imagenet/imagenet_devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
CLASS_FILEPATH = "imagenet_classes_val.txt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

###############################################################################

def normalize(img, change_to_shape=None):
    ts = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]

    if change_to_shape is not None:
        ts.insert(0, T.Resize(change_to_shape))
    

    transforms = T.Compose(ts)
    return transforms(img)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean


def save_tensor(tensor: torch.tensor, name: str, path: str = '.') -> None:
    """
    Saves a tensor to disk file.

    Args:
        tensor (torch.tensor): Tensor of interest.
        name (str): name of tensor file, extended with .pt automatically
        path (str): Path to save tensor
    """
    torch.save(tensor, f"{path}/{name}.pt")


def load_class_images_from_id(
    filepath: str,
    labels: torch.tensor,
    class_id: int | set | List,
    force_shape_to: Tuple[int, int] = (299,299)
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Loads images with the same class given a list of class indices along with their target labels.

    Args:
        filepath (str): Path to file.
        labels (Iterable[int]): List of labels in the same order as image data.
        class_id (int or list): The class ID to retrieve (50 in imagenet validaiton of each).
            If a set/list, will return the class IDs sequentially in order of increasing label ID.
        force_shape_to (Tuple[int, int], optional): Transforms the image tensors
            spatial dimensions. Defaults to (299,299).

    Returns:
        Tuple[tensor, tensor]: Tensor of (N, 3, *force_shape_to) normalized images of the same class,
            along with their labels.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File path {filepath} does not exist.")
    
    if force_shape_to and len(force_shape_to) != 2:
        raise ValueError(
            f"force_shape_to expected shape 2, got: {len(force_shape_to)}")
    
    transforms_ = []

    if force_shape_to:
        transforms_.append(T.Resize(force_shape_to))
    transforms_.append(T.ToTensor())

    transform = T.Compose(transforms_) # may want to move transforms into a Dataset
    
    class_dict = {class_id: []} if type(class_id) == int else {_id: [] for _id in class_id} 
    current_image_idx = 0
    
     # Iterate over each file in the folder
    for filename in sorted(os.listdir(filepath)):
        img_path = os.path.join(filepath, filename)
        curr_label = labels[current_image_idx].item()

        if curr_label == class_id or curr_label in class_id:
            try:
                # Open the image file
                with Image.open(img_path).convert("RGB") as img:
                    # Apply the transformations
                    img_tensor = transform(img)      
                    # print(f"[{current_image_idx}]: {img_tensor.shape}")          
                    class_dict[curr_label].append(img_tensor)
            except Exception as e:
                # if any image fails to load, STOP.
                raise RuntimeError(f"Failed to load image {filename}: {e}")
        current_image_idx += 1

    # sort keys in-order so that returned tensor is ordered based on class ID
    class_keys = sorted(class_dict.keys())
    class_dict = {k: torch.stack(class_dict[k]) for k in class_keys}
    images =  torch.stack([value for values in class_dict.values() for value in values])
    labels_idx = torch.cat([torch.tensor([key,] * len(class_dict[key])) for key in class_keys], dim=0)
    return images, labels_idx


def load_images_from_folder(
    filepath: str, 
    max_images: int = 0,
    force_shape_to: Tuple[int, int] = (299,299),
) -> List[torch.tensor]:
    """
    Loads images from folder as a list of tensors.
    TODO: Edit to batch images to avoid out of memory

    Args:
        filepath (str): Path to folder containing images (in alphanumeric order)
        max_images (int): Limits the amonut of images retrieved to the int provided.
            Mainly used for quick retrieval debugging.
        force_same_shape (Tuple[int]): Transforms the image tensors spatial dimensions
            to specified shape (H,W).
    
    Returns:
        List of (transformed) image tensors.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File path {filepath} does not exist.")
    
    if force_shape_to and len(force_shape_to) != 2:
        raise ValueError(
            f"force_shape_to expected shape 2, got: {len(force_shape_to)}")
    
    transforms_ = []

    if force_shape_to:
        transforms_.append(T.Resize(force_shape_to))
    transforms_.append(T.ToTensor())
    # Normalize when in class Dataset, but not here
    
    transform = T.Compose(transforms_)
    
    image_tensors = []
    bounded = True if max_images > 0 else False
    limit = 0

     # Iterate over each file in the folder
    for filename in sorted(os.listdir(filepath)):
        img_path = os.path.join(filepath, filename)
        try:
            # Open the image file
            with Image.open(img_path) as img:
                # Apply the transformations
                img_tensor = transform(img)                
                image_tensors.append(img_tensor)
                # print(f"[{limit}] tensor: {img_tensor}")

                # if max_images is reached, stop reading
                if bounded:
                    limit += 1
                    if limit >= max_images:
                        break
        except Exception as e:
            # if any image fails to load, STOP.
            raise RuntimeError(f"Failed to load image {filename}: {e}")

    return image_tensors


def load_labels(filepath: str) -> torch.tensor:
    """
    Loads label .txt file from filepath that contains the class idx per line.

    Args:
        filepath (str): Path to file that contains label information.

    Returns:
        torch.tensor: Tensor of shape (N,) with category indices.
            Ex. ImageNet C=1000: [1, 1000]
    """
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            # Convert the line to an integer and add it to the list
            labels.append(int(line.strip()))
    
    # Convert the list of integers to a torch.Tensor
    labels = torch.tensor(labels)
    return labels


def create_image_dataset(
    filepath: str, 
    labels_path: str,
    class_id: int | set | List = None, 
    force_shape_to: Tuple[int, int] = (299,299)
) -> ImageDataset:
    """
    Creates a dataset from the images from 'path' and labels from 'labels_path'.
    TODO: Need to change ImageDataset to access files on disk rather than take
        tensors in memory. Good news: you won't be using this function.

    Args:
        filepath (str): Path to folder containing images
        labels_path (str): Path to label file (class idx for each image, in order)
        class_id (int or list): The class ID to retrieve (50 in imagenet validaiton of each).
            If a set/list, will return the class IDs sequentially in order of increasing label ID.
        force_same_shape (Tuple[int]): Transforms the image tensors spatial dimensions
            to specified shape (H,W). If None, shapes will be maintained as they were.
            \n\tEx. (299, 299) to reshape all images to 299x299.
    """
    images = None
    labels = load_labels(labels_path)
    if class_id is None:
        images = load_images_from_folder(filepath, force_shape_to=force_shape_to)
    else:
        images, labels_idx = load_class_images_from_id(
            filepath, labels, class_id, force_shape_to=force_shape_to)

    return ImageDataset(images, labels_idx)


def create_catfish_instance(
    image1: torch.tensor,
    image2: torch.tensor,
) -> torch.tensor:
    """
    Takes image1 and image2, reshapes them to 100x100, "plasters" them onto a canvas of the
        mean color of the ImageNet dataset, ensuring no occlusion and random positions.
        Images are allowed to be batched. Random seed should be set before this to control
        its production.

    Args:
        image1 (torch.tensor): First image (1, 3, 299, 299)
        image2 (torch.tensor): Second image (1, 3, 299, 299)

    Returns:
        A 300x300 (N, 3, 299, 299) image containing the two image as described.
    """

    if len(image1.shape) > 3:
        if image1.size(0) != image2.size(0):
            raise ValueError(
                f"Batch size should be equal in both inputs: {image1.shape} vs {image2.shape}")

    # first, resize to shape 100x100
    MAX_COORD = 300

    transform = T.Compose([T.Resize((100, 100))])
    img1 = transform(image1.clone()) 
    img2 = transform(image2.clone())
    canvas = torch.stack([torch.full((300,300), mean) for mean in IMAGENET_MEAN])

    # where to place the first image
    x1 = random.randint(0, MAX_COORD - 100)
    y1 = random.randint(0, MAX_COORD - 100)
    canvas[:, x1:x1+100, y1:y1+100] = img1

    # second image depends on the first
    def subimage_leans_right(pos: int) -> bool:
        # if center of pos is on right side, else left (0)
        center = pos + 50
        return bool(max(center - 150, 0)) 

    # if room on right
    if subimage_leans_right(x1):
        x2 = random.randint(0, x1 - 100)
    else: # left
        x2 = random.randint(x1 + 100, MAX_COORD - 100)
    
    # in this case, "right" means "bottom"
    if subimage_leans_right(y1):
        y2 = random.randint(0, y1 - 100)
    else: # top side
        y2 = random.randint(y1 + 100, MAX_COORD - 100)

    canvas[:, x2:x2+100, y2:y2+100] = img2

    return canvas


def get_class_from_id(
    file_path: str,
    indices: torch.tensor
) -> Dict[int, str]:
    """
    Gets the class name for each index.
    (Given the filepath that contains each labels' class name)

    Args:
        file_path (str): Path to file with class name per line.
        indices (torch.tensor): Specific labels (corresponding to class name)

    Returns:
        Dict[int, str]: Dict of label idx and class name pairs.
    """
    result = {}
    adjusted_idx = [i - 1 for i in indices]

    # Open the file and read lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Iterate through the list of indices
        for index in adjusted_idx:
            # Ensure the index is within the range of available lines
            if 0 <= index < len(lines):
                # Split the line into components
                parts = lines[index].split()

                # Ensure the line has at least 3 parts
                if len(parts) >= 3:
                    result[(index + 1).item()] = parts[2].replace('_', '').lower()

    return result


def mix_labels(
        class_map: Dict[int, str], 
        num_labels_for_each: int, 
) -> Dict[str, Tuple[int, int]]:
    """
    Generates conjoined labels randomly given a dictionary/list of labels.
    A 'k' number of generated, joined labels will be used for each sublabel,
    defined by the 'num_labels_for_each' argument.
    Random seed to be defined outside this function.

    Args:
        class_map (Dict[int, str]): Dict that maps class indices to string label.
        num_labels_for_each (int): How many labels will be produced for each sublabel.
        random_seed (int): Seed for RNG.

    Returns:
        Dict[str, Tuple[int, int]]: Dict of {"conjoined label": [class1_idx, class2_idx]}
    """

    final_labels = {}
    combos = []
    # check if combo is unique; (ex. class 1-2 should be equal to 2-1)
    def is_unique_pair(i, j):
        return not ([i,j] in combos or [j,i] in combos)
    
    class_keys = list(class_map.keys())
    
    for idx, sublabel in class_map.items():
        for i in range(num_labels_for_each):
            pair_idx = idx
            while pair_idx == idx or not is_unique_pair(idx, pair_idx):
                pair_idx = random.choice(class_keys)

            pair = [idx, pair_idx]
            combos.append(pair) # save unique pair
            final_labels[sublabel + '-' + class_map[pair_idx]] = pair

    return final_labels


def create_catfish_dataset(
        images: torch.tensor, 
        labels: torch.tensor, 
        classes_filepath: str,
        output_path: str,
        random_seed: int = None,
) -> Tuple[DataLoader, List[str]]:
    """
    Creates a dataset analogous to the CatFish dataset in the paper 
    'Automatic Discovery of Visual Circuits' by Rajaram et al.
    New dataset contains 300x300 curated images with two downsampled and combined
    100x100 ImageNet class images, with the rest of the background being the mean
    value of the two downsampled images. Labels are conjoined for the curated image.
    (Examples: tabby-hotpot, joystick-pajama).
    NOTE: The new dataset is created in the file system at the output_path, as well
        as passed through the returned DataLoader.

    Args:
        images (torch.tensor): ImageNet images of shape (N, 3, H, W)
        labels (torch.tensor): Selected ImageNet class labels subset of shape (N,)
        classes_filepath (str): Path to class names mapping label idx to strings.
        output_path(str): Path to store outputs in ImageFolder-compatible format.
        random_seed(int): Seed for RNG.

    Returns:
        DataLoader and new class list of conjoined class categories.
    """    
    if random_seed:
        random.seed(random_seed)

    class_map = get_class_from_id(classes_filepath, labels) # {id: class} pairs

    # take combinations of 2
    new_labels = mix_labels(class_map, 2) # {newlabel: [idx pair]}

    # if catfish_labels.txt already exists, clear it
    label_output_path = os.path.join(output_path, "catfish_labels.txt")
    if os.path.exists(label_output_path):
        with open(label_output_path, 'w') as f:
            f.write('')

    # for each pair
    for i, (classname, pair) in enumerate(sorted(new_labels.items())):
        count = 0 
        # these are only 1 value so far; images need to consist of more images
        c1 = torch.where(labels == pair[0])
        c2 = torch.where(labels == pair[1])

        # create all possible catfish combinations between the two classes
        # maybe can be optimized
        catfish_imgs = []

        for img1 in images[c1]:
            for img2 in images[c2]:
                transform = T.ToPILImage() # transform to Image
                catfish_img = transform(create_catfish_instance(img1, img2))
                catfish_imgs.append(catfish_img)
                count += 1

        # record labels in one document
        for _ in range(count):
            write_labels_to_file(
                label_output_path, i)
        
        # write images
        save_images_to_directory(
            catfish_imgs,
            os.path.join(output_path, "images", classname),
            classname,
        )
    
    transforms = T.Compose([
        T.Resize((299,299)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )

    loader = torchvision.datasets.ImageFolder(
        os.path.join(output_path, "images"),
        transform=transforms,
    )

    return loader, list(sorted(new_labels.keys()))
    

def save_images_to_directory(
    images: List[Image.Image], 
    directory: str,
    prefix: str, 
    format: str ="png",
    leading_zeros: int = 8,
) -> None:
    """
    Save a list of PIL images to a specified directory with unique filenames.

    Args:
        images (): List of PIL.Image objects to be saved.
        directory (): Directory path to send images.
        prefix (): Prefix for the filenames of the saved images.
            Set to the catfish label.
        format (): Format to save the images (e.g., "JPEG", "PNG").
        leading_zeros (): # of leading zeros for image names.
    """
    # Ensure the directory exists, if not, create it
    os.makedirs(directory, exist_ok=True)
    
    # Iterate through the list of images and save them
    for i, img in enumerate(images):
        # Create a unique filename using the prefix and the index
        filename = f"{prefix}_{str(i+1).zfill(leading_zeros)}.{format.lower()}"
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            print(f"Already exists: {filepath}")
            continue
        
        # Save the image
        img.save(filepath, format=format)
        print(f"Saved: {filepath}")


def write_labels_to_file(filepath, content):
    """
    Append content to a .txt file. If the directory does not exist, it is created.

    :param filepath: The full path (including filename) of the .txt file.
    :param content: The content to append to the file.
    """
    # Ensure the directory exists, if not, create it
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Open the file in append mode and write content
    with open(filepath, 'a') as file:  # Open the file in append mode
        file.write(str(content) + '\n')  # Write the content and add a newline character
    

def idx_to_classname(
    idx: int,
    file_path: str,
    names_only: bool = False
) -> str:
    """
    Gets the class name for each index.
    (Given the filepath that contains each labels' class name)

    Args:
        file_path (str): Path to file with class name per line.
        indices (torch.tensor): Specific labels (corresponding to class name)
        names_only (bool): If file only includes classnames per line.

    Returns:
        str: Class name of label index.
    """
    result = None

    # Open the file and read lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Ensure the index is within the range of available lines
        if 0 <= idx < len(lines):
            if names_only:
                return lines[idx]
                

            # Split the line into components
            parts = lines[idx].split()

            # Ensure the line has at least 3 parts
            if len(parts) >= 3:
                result = parts[2].replace('_', '').lower()

    return result


def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X


def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X