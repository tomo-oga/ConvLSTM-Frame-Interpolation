# An Implementation of Suzuki _et. al_'s work on Frame Interpolation using ConvLSTM's and Residual Learning
 A practical implementation of _Suzuki et. al._'s work on ["Residual Learning of Video Frame Interpolation Using Convolutional LSTM"](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9145730)

![Synthesis Network](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/6287639/8948470/9145730/suzuk3-3010846-large.gif)

**Figure 1:** Overall Architecture as Described by _Suzuki et. al._




 **Note:** This is a closely followed implementation of _Suzuki Et. al_'s work, however may not be exact. Reasons for this are as follows:
 
 **In Terms of the Synthesis Network:**
 * Feature Extractions in the U-Net were not explicitely mentioned in terms of expansion ratio, or starting expansion. This led to the implmentation of the `synthesis.py` with a starting expansion of 32 channels, and an expansion ratio of 2.
 * The exact implementation of the ConvLSTM layer they used was not shown. The original implementation by _Shi et. al._ was followed.
 * Time-Series Processing of Spatio-Temporal data in layers not made to handle such data were not shown, such as for `nn.Conv2d`, `nn.Upsample`, `nn.BatchNorm2d`. This led to _batch-wise_ processing of data. While this is an effective approach, they may have used sequential processing such as in a loop to implement this, where each time-step was treated as a slice and passed through. While this was considered, it increases the time it takes for the forward pass, and therefore _batch-wise_ processing was implemented.
 * The way that the time-series data collapses to a single frame was not mentioned. Time-step channel-wise concatenation was implemented after the last `DecodingBlock` and then passed through the last two convolutional layers. This is an effective approach, however may not be the appraoch they implemented.

**In Terms of the Refinement Network:**
* The way that _I<super>'</super><sub>2.5</sub>, I<sub>2</sub>,_ and _I<sub>3</sub>_ were fed were not shown, and therefore channel-wise concatenation was implemented. Since this part of the network is meant to learn spatial and channel-wise dependencies over the global context of these three frames, this is more than likely the correct approach. This leads to 9 channels to be fed into the refinement network.
* Channel-wise attention was implemented in their work through a Squeeze-And-Excitation Network (SENet). However, the original implementation of this work, by _Hu et. al._ in ["Squeeze-and-Exciation Networks"](https://arxiv.org/abs/1709.01507) can typically be described by two fully connected layer, with a reduction ratio, denoted as _r_. This dimensional reduction, (or bottleneck) is essential for aggregating global channel interdependencies, instead of forming a one-hot activation, of certain channels. The second fully connected layer brings the representation back up to the channel-wise space. This is then multiplied channel-wise to the input features, to perform attention. However, this reduction ratio is not shown in their implementation. For this model, a reduction ratio of 3 is implemented. As in `r = 3`.

### Structure
* `blocks.py` holds all essential blocks creating the network. This includes the `EncodingBlock`, `DecodingBlock`, `ChannelAttention`, `SpatialAttention`, and `AttentionBlock`s.
* `convlstm.py` holds the `ConvLSTMCell` that becomes implemented in the `ConvLSTMLayer` which unrolls the cell over time-steps.
* `synthesis.py` takes the `EncodingBlock` and `DecodingBlock` and creates the U-Net architecture as described in the paper, following an initial expansion of 32 channels, and an expansion ratio of 2 thereafter, which should learn to map the arbitrary residual function between the ground truth, and linear interpolation, by feeding in 4 frames. 
* `refinement.py` takes the `AttentionBlock` (which implements the `ChannelAttention` and `SpatialAttention`) to form the residual refinement network.
* `net.py` contains the trainable `Net` which implements the `SynthesisNet` and `RefinementNet` together. `Net` will return a tuple with the synthesis frame and refined frame for training. This file also contains `loss_fn` which given the synthesis, refined, and ground truth frame, will return the loss for this model as described in the paper.
