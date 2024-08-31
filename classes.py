import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Union, Dict
from math import ceil, floor
import random
import utils
import sys
import heapq

import time
from functools import wraps

# Dataset to use with ImageNet
# TODO: Retrieve files to __get_item__ rather than store all at once (runs out of memory)
class ImageDataset(Dataset):
    def __init__(self, images, labels, normalize=True):
        self.images = images
        self.labels = labels

        if normalize:
            self.transform = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
        else:
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Takes in a class and computes feature visualizations
# for neurons, channels, and layers of CNN-based models
class ModelVisualizer:
    def __init__(self,
                 model: torch.nn.Module,
                 input_size: Tuple[int, int] = (299, 299),
                 device: str = 'cpu'):
        """
        Initialize the visualizer with the model and optional dataset.
        """
        self.model = model.to(device)
        self.input_size = input_size
        self.device = device

    def visualize_activations(self,
        target_layer: str,
        target_channel: int = None,
        target_neuron = None, # remove later
        input_image = None,
        lr=25.,
        l2_reg=1e-4,
        max_iter=500,
        verbose_iters=50,
        **kwargs):
        """
        Perform gradient ascent to maximize activations of the specified channel.
        """

        blur_every = kwargs.pop('blur_every', 25)
        max_jitter = kwargs.pop('max_jitter', 16)

        if input_image is not None:
            img = input_image.clone().unsqueeze(0).to(self.device)
        else:
            # Initialize a random input image
            img = torch.randn((1, 3, *self.input_size), device=self.device)

        img.requires_grad_()
        print(img.device)

        for iteration in range(max_iter):
            # add tranformation robustness [jitter] for better output visual
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.data.copy_(utils.jitter(img.data, ox, oy))

            img, step_activation = gradient_ascent_step(
                img=img,
                model=self.model,
                target_layer=target_layer,
                target_channel=target_channel,
                target_neuron=target_neuron,
                learning_rate=lr,
                l2_reg=l2_reg,
            )

            def show_img(img):
                tfm = transforms.ToPILImage()
                _img = tfm(img)
                _img.show()

            img.data.copy_(utils.jitter(img.data, -ox, -oy)) # undo jitter after step

            # As regularizer, clamp and periodically blur the image for better image output
            # from EECS598 assign6
            for c in range(3):
                lo = float(-utils.IMAGENET_MEAN[c] / utils.IMAGENET_STD[c])
                hi = float((1.0 - utils.IMAGENET_MEAN[c]) / utils.IMAGENET_STD[c])
                img.data[:, c].clamp_(min=lo, max=hi)
            if iteration % blur_every == 0:
                utils.blur_image(img.data, sigma=0.5)

            # output images periodically
            if (verbose_iters is not None and iteration % verbose_iters == 0) or iteration == max_iter - 1:
                show_img(utils.denormalize(img.squeeze(0)))
                print("iteration: ", iteration)
                print("activation: ", step_activation)

        return img

    def get_maximum_activations(self,
            dataset: Union[Dataset, torch.tensor],
            topk: int,
            target_layer: str,
            target_channel: int = None,
            target_neuron: int = None
        ) -> List[torch.tensor]:
        """
        Returns the top 'k' images/tensors from a dataset that maximally activates
        a specified target_layer, channel, or neuron.

        Args:
            dataset (DataLoader | torch.tensor): Dataset of images.
            topk (int): How many images to return.
            target_layer (str): The name of the model layer
            target_channel (int, optional): Index of channel. Defaults to None.
            target_neuron (int, optional): Index of neuron. Defaults to None.

        Returns:
            List[torch.tensor]: List of "images" that maximally activate
            the selected part of the model.
        """
        activations = torch.zeros(len(dataset))

        def get_activation(name):
            def hook(model, input, output):
                A = output.detach()
                H,W = A.shape[-2:]
                # print("untouched shape", output.shape)
                if target_channel is not None:
                    A = A[:, target_channel, :, :]
                    # print("channel shape", activations[target_layer].shape)
                if target_neuron is not None:
                    A = A[..., target_neuron // H, target_neuron % W] # hardcoded for quickness
                # print("shape", A.shape)
                activations[name] = torch.sum(A)
            return hook

        hook_target = getattr(self.model, target_layer, None)
        if hook_target is None:
            raise ValueError(f"target_layer '{target_layer}' does not exist in the model.")


        for i, (img, label) in enumerate(dataset):
            handle = hook_target.register_forward_hook(get_activation(i))
            a = self.model(img.unsqueeze(0).to(self.device))
            handle.remove()

        # print(activations)
        val, idx = activations.topk(k=topk, sorted=True)
        return idx, val


# Main class that discovers circuits inside a pretrained model.
class CircuitExtractor():
    def __init__(self,
        model: nn.Module,
        input_distrib: torch.tensor,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.input = input_distrib.to(device) # shape (N, 3, 299, 299)
        self.device = device
        self.map_to_name = {}
        self.map_to_idx  = {} # inverse of above, for convenience
        self.layer_shapes = {} 

        # when build_circuit gets int 'start/end' values, can convert to layername
        # and vice versa (though, not efficiently -- can be improved on)
        for i, (name, _) in enumerate(model.named_children()):
            self.map_to_name[i] = name
            self.map_to_idx[name] = i

        self.model.eval() # in case it hasn't been set before


    def __call__(self, *args, **kwargs):
        return self.build_circuit(*args, **kwargs)


    def build_circuit(self,
        start: Union[str, int],
        end: Union[str, int],
        k: Union[List[int], torch.tensor, int]
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Builds a circuit from the 'start' layer to the 'end' layer.
        For each layer, it takes the 'k' neurons that led to the maximal attribution sum.

        Args:
            start (str or int): Layer of model to start circuit discovery (index or name).
            end (str or int): Layer of model to end circuit discovery (index or name).
            k (List[int] or tensor): # neurons to include in the circuit for each layer.

        Returns:
            A circuit represented as a List of Lists/Tensors (list of 'k' neurons per element)
        """
        if not self._valid_range(start, end):
            raise ValueError(f"Starting layer ({start}) should come before ending layer ({end}).")

        self.circuit = {}  # Dict of list of neuron indices (int)
        self.start = self._idxof(start)
        self.end = self._idxof(end)
        self.k = k

        # circuit[L] initialization
        # _, topk_n_neurons = self.topk_attribution_scores_online(self.end - 1, self.end)
        _, topk_n = self.get_topk_neurons(self.end - 1, self.end, max_dim='init')
        torch.cuda.empty_cache()
        self.circuit[self._nameof(end)] = topk_n.clone().detach()

        # backwards pass
        self.backward_pass()

        # refine until convergence (backward + forward loop)
        self.refine_circuit(max_iters=0)

        return self.circuit
    
    
    def refine_circuit(self, max_iters=0):
        """
        Loops backward and forward passes following CLA until convergence.
        When max_iters is 0 or negative, it runs indefinitely until convergence. 
        """
        if not hasattr(self, 'start') or not hasattr(self, 'end') or not hasattr(self, 'circuit'):
            RuntimeError("build_circuit hasn't been called defining 'start' and 'end' layers yet. "
                         "Cannot run refinement until running 'build_circuit'.")
            
        prev_circuit = {}
        iters = 0
        while not self._check_circuit_equality(self.circuit, prev_circuit):
            print("CURRENT CIRCUIT: ", self.circuit, "\nPREVIOUS: ", prev_circuit)
            prev_circuit = self.circuit.copy()
            self.backward_pass()
            self.forward_pass()

            if max_iters != 0 and iters <= max_iters:
                if iters == max_iters:
                    break
                iters += 1 
                
        print("FINAL CIRCUIT: ", self.circuit, "\nPREVIOUS: ", prev_circuit)

    def backward_pass(self):
        """ Runs through a backward pass of the CLA algorithm. """
        if not hasattr(self, 'start') or not hasattr(self, 'end') or not hasattr(self, 'circuit'):
            RuntimeError("build_circuit hasn't been called defining 'start' and 'end' layers yet. "
                         "Cannot run a backwards pass until running 'build_circuit'.")
        
        for i in range(self.end - 1, self.start - 1, -1): # L - 1 to start
            print("backward", i)
            _, topk_m = self.get_topk_neurons(i, i + 1, max_dim='m')            
            torch.cuda.empty_cache()
            self.circuit[self._nameof(i)] = topk_m.clone().detach()

    def forward_pass(self):
        """ Runs through a forward pass of the CLA algorithm. """
        if not hasattr(self, 'start') or not hasattr(self, 'end') or not hasattr(self, 'circuit'):
            RuntimeError("build_circuit hasn't been called defining 'start' and 'end' layers yet. "
                         "Cannot run forward pass until running 'build_circuit'.")

        for i in range(self.start + 1, self.end + 1):
            print("forward", i)
            _, topk_n = self.get_topk_neurons(i - 1, i, max_dim='n')
            torch.cuda.empty_cache()
            self.circuit[self._nameof(i)] = topk_n.clone().detach()
    
    def get_topk_neurons(self, l1, l2, max_dim) -> Tuple[Tuple[float], Tuple[int]]:
        """
        Get the topk 'm' or 'n' neurons of an attribution matrix.
        The approach used in this method is GPU-memory adaptive by loading in chunks at a time.

        Args:
            l1 (int): Index of the first layer.
            l2 (int): Index of the second layer.
            max_dim (str): Which dimension to take the max from.
                If 'n' or 'm', takes the max of the sum, as seen in the paper.
                If 'init', computes the max 'n' neurons over all 'm' instead of a sum.

        Raises:
            ValueError: If build_circuit was not called previously.

        Returns:
            Tensors of topk neurons' values and indices.
        """
        
        # exception checks here:
        if not hasattr(self, 'k'): # check if build_circuit has run
            RuntimeError("build_circuit hasn't been called with parameter 'k' yet. \
                         Cannot compute circuit neurons for the current layer.")
        print("\nRUNNING with k:", self.k, " dim_type=", max_dim)

        if max_dim not in {'n', 'm', 'init'}:
            raise ValueError("max_dim argument needs to be 'n' or 'm' or 'init'")
        
        # compute GPU memory to use 
        num_chunks, l1_size, l2_size = self._compute_chunk_size(
            l1, l2,  # these layers are ALWAYS in chronological order
            limit=torch.cuda.mem_get_info(0)[0], # all GPU memory available in gpu:0 [assuming 1 GPU]
            max_dim=max_dim
        )

        print("memcheck", torch.cuda.mem_get_info()[0])

        print(f"l1size: {l1_size}, l2size: {l2_size}")

        chunk_size = ceil(l2_size / num_chunks) # size (of floats) per chunk
        buffer = torch.zeros((chunk_size, l1_size), device=self.device) # holds 1 chunk of attribution scores

        print(f"chunk_size: {chunk_size} x{num_chunks}")
        print(buffer.shape)

        topk_neurons = [] # list of neurons for circuit
        chunk_sums = None  # if max_dim == 'init'
        chunk_idx = 0 # for chunk_sums and chunk_topk (below)

        # if not 'init', then we're summing over a dimension of the attribution matrix
        if max_dim in {'n', 'm'}: 
            chunk_sums = torch.zeros(
                (num_chunks, l1_size if max_dim == 'm' else chunk_size),
                device=self.device
            )

        # start processing chunks of the Jacobian => attribution matrix
        for i in range(num_chunks):
            print("chunk#", i)
            buffer.zero_() # clear buffer every chunk
            idx_slice = slice(
                i * chunk_size,
                (i+1) * chunk_size if i < num_chunks - 1 else l2_size # if last chunk size != the others
            )

            # get the attribution scores for the current chunk (ugly parametrization)
            buffer = self.compute_attribution_score_chunk(l1, l2, max_dim, buffer, idx_slice, chunk_size, l1_size, l2_size)

            if chunk_sums is not None: # if max_dim is 'n' or 'm'
                if max_dim == 'm':
                    # don't need to select from circuit if 'm'; already does it in compute_attribution_score_chunk
                    chunk_sum = buffer.sum(dim=0) # 'm': (1, m) => (n, m) when finished
                if max_dim == 'n': # we need to only consider the selected topk 'm' neurons
                    sel_buffer = buffer[:, self.circuit[self._nameof(l1)]]
                    chunk_sum = sel_buffer.sum(dim=1).t() # 'n': (n, 1) => ()

                chunk_sums[chunk_idx:chunk_idx+1] = chunk_sum
                chunk_idx += 1
            else: # 'init': treat 'topk_neurons' as a heap and evaluate global topk on the fly
                k_val, k_idx = buffer.max(dim=1)[0].topk(self.k)
                k_idx = k_idx + (i * chunk_size) # global index

                for k, val in enumerate(k_val):
                    if len(topk_neurons) < self.k:
                        heapq.heappush(topk_neurons, (val, k_idx[k]))
                    elif val > topk_neurons[0][0]:  # If the current number is larger than the smallest in the heap
                        # Replace the smallest element with the current number
                        heapq.heapreplace(topk_neurons, (val, k_idx[k]))
               
        del buffer

        # for 'init': topk_neurons already contains the max topk 'n' neurons
        if max_dim != 'init':
            # at this point all chunk sums are computed; now, get the global sum for all chunks
            # (if max_dim == 'n', then this global sum is already computed)
            topk_candidates = chunk_sums.reshape(-1) if max_dim == 'n' else chunk_sums.sum(dim=0) # (n,) or (m,)
            k_val, k_idx = topk_candidates.topk(self.k, sorted=False)
            topk_neurons = [(k_val[k], k_idx[k]) for k in range(len(k_val))]
        
        final_val, final_idx = zip(*sorted(topk_neurons, reverse=True, key=lambda k: k[0])) # sort here
        return torch.tensor(final_val, device=self.device), torch.tensor(final_idx, device=self.device)
        

    def compute_attribution_score_chunk(self, l1, l2, max_dim, buffer, idx_slice, chunk_size, layer_size, next_size):
        N = self.input.size(0)

        if max_dim == 'm':
            # if 'm', selects the 'm' neurons that maximize the summed attrbution score of topk 'n' neurons
            selected_neurons = self.circuit[self._nameof(l2)]
            rows = get_identity_subset(next_size, selected_neurons).to(self.device)
        elif max_dim == 'n':
            # if 'n', need to compute full Jacobian again; get 'n' neurons that max the sum the topk 'm' neurons
            selected_neurons = self.circuit[self._nameof(l1)]
            rows = get_identity_subset(next_size, idx_slice.start, idx_slice.stop).to(self.device)
        else:
            # if 'init', select the 'n' neurons with the largest attribution to any 'm' neuron
            selected_neurons = None
            rows = get_identity_subset(next_size, idx_slice.start, idx_slice.stop).to(self.device) # chunked(n) x m sized
        
        print(f"selected neurons '{'n' if max_dim == 'm' else 'm'}':", selected_neurons)

        # for all input distribution samples
        for inst in range(N):
            activations = self._get_activations(l1, l2, inst=inst, l2_norm=True)
            l1_act = activations[self._nameof(l1)]
            l2_act = activations[self._nameof(l2)]

            # function for single instance gradient computation
            # (used alongside vmap for partial vectorization along chunk dimension)
            def get_grad_for_neuron(n):
                return torch.autograd.grad(
                    l2_act.view(1, -1).pow(2).sqrt(), # output (w/ l2 norm)
                    l1_act, # input
                    n.unsqueeze(0), # vector in vec-Jacobian product
                    retain_graph=True if inst < N - 1 else False)[0]

            # vectorize the chunk to get a subset of rows of the Jacobian matrix
            partial_J = torch.vmap(get_grad_for_neuron)(rows)

            # add the averaged sum to the buffer incrementally, then convert into attribution matrix
            if partial_J.size(0) == chunk_size:
                attr = partial_J.view(chunk_size, layer_size) * torch.abs(l1_act.view(-1))
                buffer = attr.detach() / N
            else:
                attr = partial_J.view(partial_J.size(0), layer_size) * torch.abs(l1_act.view(-1))
                buffer[0:partial_J.size(0)] += attr.detach() / N

            del partial_J, attr

        return buffer
    

    def get_attribution_scores(self, start=None, end=None):
        """
        Computes the attribution scores of all layers between start and end layers.

        NOTE: Heavily memory intensive; for a single inception4b layer, need ~40GB of memory to store.
        This is because we need to store the Jacobian between layers [shape: (512*14*14)x(512*14*14)]
        Therefore, DO NOT USE this unless you have the memory to do so.
        """

        if start is None or end is None: # if ran in "internal mode"
            # if start and end weren't defined in build_circuit, don't run
            if not hasattr(self, 'start') or not hasattr(self, 'end'):
                RuntimeError("build_circuit hasn't been called with 'start' and 'end' parameters yet. "
                             "Cannot compute attribution scores.")
            else: # if called by build circuit, use these instead
                start = self.start
                end = self.end

        # Compute attribution matrices:
        attributions = {}
        activations = self._get_activations(start, end) # calls forward pass inside
        N = self.input.size(0)

        # iterate through all activations (in reverse)
        for layer_idx in range(self._idxof(end) - 1, self._idxof(start) - 1, -1):
        # iterate through batch layer, accumulate attributions (average after all iterations completed)
            print("attr for ", self._nameof(layer_idx))
            for i in range(N):
                print("iter#", i)
                # activations are stored as tensors
                curr_activs = activations[self._nameof(layer_idx)][i] # (C, H, W)_i
                next_activs = activations[self._nameof(layer_idx + 1)][i] # (C, H, W)_j where j=i+1
                C1, H1, W1 = curr_activs.shape
                C2, H2, W2 = next_activs.shape
                print(curr_activs.shape)
                print(next_activs.shape)

                def layer_func(x):
                    # get the l2 norm of each neuron activation (scalar; equivalent to absolute value)
                    next_out = (self._get_layer(layer_idx + 1)(x)).squeeze(0).pow(2).sqrt() # l2 norm
                    return next_out

                # J shape: (Cj, Hj, Wj, Ci, Hi, Wi)
                jacobian = torch.func.jacrev(layer_func)(curr_activs.unsqueeze(0))

                # (1, CHWi) * (CHWj, CHWi) broadcasts to (CHWj, CHWi)
                attr = torch.abs(curr_activs.view(-1, 1)) * jacobian.view(C1*H1*W1, C2*H2*W2)
                if self._nameof(layer_idx) not in attributions:
                    attributions[self._nameof(layer_idx)] = torch.zeros((C2*H2*W2, C1*H1*W1), device=curr_activs.device)
                attributions[self._nameof(layer_idx)] += attr.t() / N # moving average

                del attr, jacobian # remove tensors that are no longer needed

            del next_activs # no longer need this activation matrix

        return attributions


    def prune_circuit(self):
        # TODO later
        pass


    def prune_edge(self):
        # TODO later
        pass


    def layer_names(self):
        return list(self.map_to_name.values())


    def _get_activations(self, start, end, inst: int = None, l2_norm=True):
        activations = {} # in the form {"layer_name": its activations}
        hooks = []

        def get_activation_for(layername):
            name = self._nameof(layername)
            def hook(model, input, output):
                activations[name] = output # activation here
                # print(f"Layer Activation shape for {name}:", output.shape)
            return hook

        # if end is before start (forward mode), then need to adjust for this
        bound1, bound2 = self._idxof(start), self._idxof(end)
        # attach hook for all intermediate layers (inclusive)
        for layer_idx in range(min(bound1, bound2), max(bound1, bound2) + 1):
            curr_layer = self._get_layer(layer_idx)
            h = curr_layer.register_forward_hook(get_activation_for(layer_idx))
            hooks.append(h)

        if inst is not None: # if we want a SINGLE instance pass
            _ = self.model(self.input[inst].unsqueeze(0))
        else:
            _ = self.model(self.input) # forward pass to get activations

        # no longer need hooks
        for h in hooks:
            h.remove()

        return activations


    def _compute_chunk_size(self,
        layer1: int,
        layer2: int,
        limit: int,
        max_dim: str,
    ) -> int:
        """
        Helper function for Circuit Extractor; calculating how much memory a Jacobian computation would use,
        then finding the number of chunks that would maximize performance (splitting the Jacobian into VJP products
        of different rows [vectorized]) while adhering to the memory constraint (in bytes).

        Args:
            layer1 (torch.tensor): First layer (can be before or after 'layer2' within the model)
            layer2 (torch.tensor): Second layer (if after 'layer1', this is backward, otherwise forward)
            forward (bool): DEPRECATED.
            limit (int): Memory constraint in bytes. If using torch, this would be torch.cuda.mem_get_info()[0].
            leave_overhead(bool): calculate how much memory to preserve for external tensor creation.
                (when summed=True in attribution score computation)
            batched (bool): If the incoming tensors have a batched dimension, it is omitted.

        Returns:
            Number of chunks to split the Jacobian computation into (parallel VJPs), and BOTH layer sizes.
        """
        bytes_dict = bytes_in_layers(self.model, input_shape=(1, *self.input.shape[1:]), device=self.device)

        L1 = bytes_dict[self._nameof(layer1)] # neurons 'm'
        L2 = bytes_dict[self._nameof(layer2)] # neurons 'n'

        if max_dim == 'm':
            # easier since we only need to compute 'self.k' 'n' neurons w.r.t 'm'
            limit = floor(limit * 0.95)
        else:
            # could be a better way, but need to half to create 'rows' for vectorized vJP
            limit = floor(limit * 0.48)

        # if not enough bytes for a single row of 'm' neurons per 'n' (assuming backward, but works both ways)
        if L1 > limit:
            raise RuntimeError(f"Not enough memory to compute chunks necessary "
                               f"for attribution scores/build_circuit."
                               f"\nBytes available: {limit}, bytes requested: {L1}")
        
        chunk_size = floor(limit / L1) # how many rows of 'n' can be computed at once
        print("MAX CHUNK SIZE:", chunk_size)

        # convert to num floats (// 4)
        L1 = L1 // self.input.element_size()
        L2 = L2 // self.input.element_size()
        num_chunks = ceil(L2 / chunk_size)

        return num_chunks, L1, L2
    

    def _check_circuit_equality(self, c1: Dict[str, torch.Tensor], c2: Dict[str, torch.Tensor]):
        layer_names = self.layer_names() # these are in order
        if len(c1) != len(c2):
            return False
        
        for name in layer_names:
            if name not in c1 or name not in c2:
                continue
            c1_neurons = c1[name]
            c2_neurons = c2[name]
    
            if len(c1_neurons) != len(c2_neurons):
                return False

            # evaluating elements of a tensor by their difference; 
            if not torch.all(torch.sort(c1_neurons)[0] == torch.sort(c2_neurons)[0]):
                return False
            
        return True


    def _get_layer(self, layername: Union[str, int]) -> nn.Module:
        return getattr(
            self.model,
            self._nameof(layername))


    def _valid_range(self, layer1: Union[str, int], layer2: Union[str, int]) -> bool:
        """ Only returns true if 1st layer argument comes before 2nd in the model. """
        return self._circuit_len(layer1, layer2) > 0


    def _circuit_len(self, start: Union[int, str], end: Union[int, str]) -> int:
        """ Length of the circuit (by number of layers it passes through) """
        return self._idxof(end) - self._idxof(start)


    # helpers for convenience when name or idx is better suited to be used
    def _nameof(self, id_: Union[str, int]) -> str:
        """ Gets the name of a layer. If already a string, returns it. """
        if isinstance(id_, str):
            return id_
        else:
            return self.map_to_name[id_]

    def _idxof(self, id_: Union[str, int]) -> int:
        """ Gets the index of a layer. If already an integer, returns it. """
        if isinstance(id_, int):
            return id_
        else:
            return self.map_to_idx[id_]


########################### HELPER FUNCTIONS BELOW #############################


def gradient_ascent_step(
    img: torch.tensor,
    model: nn.Module,
    target_layer: str,
    target_channel: int = None,
    target_neuron: int = None,
    **kwargs
) -> torch.tensor:
    """
    Performs gradient step update to generate an image that amplifies the
    activation of a layer/neuron in a model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_layer: Layer that we want the image to maximize activations for.
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - neuron_idx: Which neuron within the layer to maximally activate.
        If None, applies for the whole layer.
    - learning_rate: How big of a step to take

    Outputs:
    - optimized image, activation of the neuron in this step
    """
    img.requires_grad_()
    l2_reg = kwargs.pop("l2_reg", 1e-4)
    learning_rate = kwargs.pop("learning_rate", 25.)

    activations = {}

    # create hook to capture activations
    def hook(model, input, output):
        activations[target_layer] = output.clone()
        H, W = output.shape[-2:]

        # print("untouched shape", output.shape)
        if target_channel is not None:
            activations[target_layer] = (activations[target_layer])[:, target_channel, :, :]
            # print("channel shape", activations[target_layer].shape)
        if target_neuron is not None:
            if target_neuron >= H * W or target_neuron < 0:
                raise ValueError(f"target_neuron is out of range of the chosen layer (0 to {H * W - 1}).")
            neuron_row = target_neuron // H
            neuron_col = target_neuron % W
            activations[target_layer] = (activations[target_layer])[..., neuron_row, neuron_col]

    # hook layer to get activations froma
    hook_target = getattr(model, target_layer, None)
    if hook_target is None:
        raise ValueError(f"target_layer '{target_layer}' does not exist in the model.")
    handle = hook_target.register_forward_hook(hook)

    _ = model(img) # run forward pass

    # l2 regularization for better output image
    loss = torch.sum(activations[target_layer]) - l2_reg * torch.sum(img ** 2)
    loss.backward()

    grad = img.grad
    grad /= grad.norm() + 1e-8

    with torch.no_grad():
        # gradient ascent (+ instead of -)
        img.data += learning_rate * grad
        img.grad.zero_()

    handle.remove()

    return img, torch.sum(activations[target_layer])


def size_in_bytes(tensor: torch.tensor):
    return tensor.element_size() * tensor.nelement()


def bytes_in_layers(model, input_shape, device='cpu'):
    """ 
    For each layer, returns shape and size of layer
    (in bytes) by running a dummy forward pass. 
    """
    x = torch.zeros(input_shape, device=device)
    shapes = {}

    with torch.no_grad():
        for name, layer in model.named_children():
            if isinstance(layer, nn.Module) and name != "":
                x = layer(x)  # Pass the tensor through the layer

                # Handle specific cases where flattening is required before linear layers
                if isinstance(layer, nn.AdaptiveAvgPool2d) or isinstance(layer, nn.Flatten):
                    x = torch.flatten(x, 1)
                elif isinstance(layer, nn.Linear):
                    x = x.view(x.size(0), -1)  # Ensure the correct shape for the linear layer

                shapes[name] = size_in_bytes(x)

    return shapes


def get_identity_subset(n, start_row, end_row=None):
    """
    Get a subset of columns from a large identity matrix.
    
    Args:
    n (int): The size of the (virtual) identity matrix.
    start_col (List/tensor/int): The starting column index (inclusive).
        IF start_row is a list, we instead select rows per index.
        (All elements must be smaller than 'n', and end_row will be ignored)
    end_col (int): The ending column index (exclusive), used if start_row is an index.
    
    Returns:
    torch.Tensor: A subset of the identity matrix of shape (len(row_elements), n)
    """
    if end_row is None:
        result = torch.zeros(len(start_row), n)
        result[torch.arange(len(start_row)), start_row] = 1
        return result
    
    num_rows = end_row - start_row
    result = torch.zeros(num_rows, n)
    
    # Calculate the range of rows that will have 1s
    col_indices = torch.arange(start_row, end_row)
    
    # Only set 1s where row index equals column index in the original matrix
    result[torch.arange(num_rows), col_indices] = 1
    
    return result