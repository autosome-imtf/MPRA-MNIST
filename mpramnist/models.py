import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from collections import OrderedDict


############# Legnet############
def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (
            b,
            c,
            _,
        ) = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    def __init__(
        self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor

        block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.inner_dim,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=ks,
                groups=self.inner_dim,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            SELayer(self.inner_dim, reduction=self.se_reduction),
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.in_ch,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.in_ch),
            activation(),
        )

        self.block = block

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                kernel_size=self.ks,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.out_ch),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(
                in_channels=in_features, out_channels=out_features, kernel_size=1
            ),
        )

    def forward(self, x):
        return self.block(x)


class HumanLegNet(nn.Module):
    def __init__(
        self,
        in_ch,
        output_dim,
        stem_ch=64,
        stem_ks=11,
        ef_ks=9,
        ef_block_sizes=[80, 96, 112, 128],
        pool_sizes=[2, 2, 2, 2],
        resize_factor=4,
        activation=nn.SiLU,
    ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)

        self.in_ch = in_ch
        self.stem = LocalBlock(
            in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation
        )

        blocks = []
        self.output_dim = output_dim
        in_ch = stem_ch
        out_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation,
                    )
                ),
                LocalBlock(
                    in_ch=in_ch * 2, out_ch=out_ch, ks=ef_ks, activation=activation
                ),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity(),
            )
            in_ch = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)

        self.mapper = MapperBlock(in_features=out_ch, out_features=out_ch * 2)
        self.head = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch * 2),
            nn.BatchNorm1d(out_ch * 2),
            activation(),
            nn.Linear(out_ch * 2, self.output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

################## MPRAnn ###########################

class MPRAnn(nn.Module):
    def __init__(self, output_dim, end_sigmoid: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=250, kernel_size=7, padding="valid")
        self.bn1 = nn.BatchNorm1d(250)
        self.conv2 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=8, padding="valid")
        self.bn2 = nn.BatchNorm1d(250)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(in_channels=250, out_channels=250, kernel_size=3, padding="valid")
        self.bn3 = nn.BatchNorm1d(250)
        self.conv4 = nn.Conv1d(in_channels=250, out_channels=100, kernel_size=2, padding="valid")
        self.bn4 = nn.BatchNorm1d(100)
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1) 
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.LazyLinear(300)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(300, self.output_dim)
        self.end_sigmoid = end_sigmoid



    def forward(self, seq):
        #seq = seq.permute(0, 2, 1)
        seq = self.conv1(seq)
        seq = F.relu(seq)
        seq = self.bn1(seq)
        seq = self.conv2(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn2(seq)
        seq = self.pool1(seq)
        seq = self.dropout1(seq)
        seq = self.conv3(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn3(seq)
        seq = self.conv4(seq)
        seq = F.softmax(seq, dim=1)
        seq = self.bn4(seq)
        seq = self.pool2(seq)
        seq = self.dropout2(seq)
        seq = seq.reshape((seq.shape[0], -1))
        seq = self.fc1(seq)
        seq = F.sigmoid(seq)
        seq = self.dropout3(seq)
        seq = self.fc2(seq)
        if self.end_sigmoid:
            seq = F.sigmoid(seq)
        seq = seq.squeeze(-1)

        return seq

################## Small classification Net ####################


class CNN_cls(nn.Module):
    def __init__(
        self, seq_len, block_sizes=[16, 24, 32, 40, 48], out_ch=64, kernel_size=3
    ):
        super().__init__()
        self.block_sizes = block_sizes
        self.seq_len = seq_len
        self.out_ch = out_ch
        nn_blocks = []

        for in_bs, out_bs in zip([4] + block_sizes, block_sizes):
            block = nn.Sequential(
                nn.Conv1d(
                    in_bs, out_bs, kernel_size=kernel_size, padding=kernel_size // 2
                ),
                nn.SiLU(),
                nn.BatchNorm1d(out_bs),
                nn.Dropout(0.3),
            )
            nn_blocks.append(block)

        final_feature_size = seq_len

        self.conv_net = nn.Sequential(
            *nn_blocks,
            nn.Flatten(),
        )
        self.after_conv = nn.Sequential(
            nn.Linear(block_sizes[-1] * final_feature_size, self.out_ch),
            nn.SiLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(self.out_ch, self.out_ch),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(self.out_ch),
            nn.Linear(self.out_ch, 1),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.after_conv(x)
        out = self.head(x).squeeze()

        return out


######################## DeepSTARR #########################


class DeepStarr(nn.Module):
    """DeepSTARR model from de Almeida et al., 2022;
    see <https://www.nature.com/articles/s41588-022-01048-5>
    """

    def __init__(
        self,
        output_dim,
        d=256,
        conv1_filters=None,
        learn_conv1_filters=True,
        conv2_filters=None,
        learn_conv2_filters=True,
        conv3_filters=None,
        learn_conv3_filters=True,
        conv4_filters=None,
        learn_conv4_filters=True,
    ):
        super().__init__()

        if d != 256:
            print(
                "NB: number of first-layer convolutional filters in original DeepSTARR model is 256; current number of first-layer convolutional filters is not set to 256"
            )

        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters

        assert not (conv1_filters is None and not learn_conv1_filters), (
            "initial conv1_filters cannot be set to None while learn_conv1_filters is set to False"
        )
        assert not (conv2_filters is None and not learn_conv2_filters), (
            "initial conv2_filters cannot be set to None while learn_conv2_filters is set to False"
        )
        assert not (conv3_filters is None and not learn_conv3_filters), (
            "initial conv3_filters cannot be set to None while learn_conv3_filters is set to False"
        )
        assert not (conv4_filters is None and not learn_conv4_filters), (
            "initial conv4_filters cannot be set to None while learn_conv4_filters is set to False"
        )

        # Layer 1 (convolutional), constituent parts
        if conv1_filters is not None:
            if (
                learn_conv1_filters
            ):  # continue modifying existing conv1_filters through learning
                self.conv1_filters = nn.Parameter(torch.Tensor(conv1_filters))
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = (
            nn.ReLU()
        )  # name the first-layer activation function for hook purposes
        self.maxpool1 = nn.MaxPool1d(2)

        # Layer 2 (convolutional), constituent parts
        if conv2_filters is not None:
            if (
                learn_conv2_filters
            ):  # continue modifying existing conv2_filters through learning
                self.conv2_filters = nn.Parameter(torch.Tensor(conv2_filters))
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)

        # Layer 3 (convolutional), constituent parts
        if conv3_filters is not None:
            if (
                learn_conv3_filters
            ):  # continue modifying existing conv3_filters through learning
                self.conv3_filters = nn.Parameter(torch.Tensor(conv3_filters))
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)

        # Layer 4 (convolutional), constituent parts
        if conv4_filters is not None:
            if (
                learn_conv4_filters
            ):  # continue modifying existing conv4_filters through learning
                self.conv4_filters = nn.Parameter(torch.Tensor(conv4_filters))
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)

        # Layer 5 (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)

        # Layer 6 (fully connected), constituent parts
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)

        # Output layer (fully connected), constituent parts
        self.fc7 = nn.Linear(256, output_dim)

    def get_which_conv_layers_transferred(self):
        layers = []
        if self.init_conv1_filters is not None:
            layers.append(1)
        if self.init_conv2_filters is not None:
            layers.append(2)
        if self.init_conv3_filters is not None:
            layers.append(3)
        if self.init_conv4_filters is not None:
            layers.append(4)
        return layers

    def forward(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)

        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)

        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)

        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool4(cnn)

        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Output layer
        y_pred = self.fc7(cnn)

        return y_pred


############################# Basset branched for Malinois ##############################


class Conv1dNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        batch_norm=True,
        weight_norm=True,
    ):
        super(Conv1dNorm, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_channels,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.conv(input))
        except AttributeError:
            return self.conv(input)


class LinearNorm(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, batch_norm=True, weight_norm=True
    ):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(
                out_features,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )

    def forward(self, input):
        try:
            return self.bn_layer(self.linear(input))
        except AttributeError:
            return self.linear(input)


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups

        # initialize weights
        self.weight = torch.nn.Parameter(
            torch.zeros(groups, in_group_size, out_group_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(groups, 1, out_group_size))

        # change weights to kaiming
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        reorg = (
            x.permute(1, 0)
            .reshape(self.groups, self.in_group_size, -1)
            .permute(0, 2, 1)
        )
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = (
            hook.permute(0, 2, 1)
            .reshape(self.out_group_size * self.groups, -1)
            .permute(1, 0)
        )

        return reorg


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_group_size,
        out_group_size,
        n_branches=1,
        n_layers=1,
        activation="ReLU",
        dropout_p=0.5,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers = n_layers

        self.branches = OrderedDict()

        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)

        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features

        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(
                    self,
                    f"branched_layer_{i + 1}",
                    GroupedLinear(cur_size, out_group_size, n_branches),
                )
            else:
                setattr(
                    self,
                    f"branched_layer_{i + 1}",
                    GroupedLinear(cur_size, hidden_group_size, n_branches),
                )
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)

        i = -1
        for i in range(self.n_layers - 1):
            hook = getattr(self, f"branched_layer_{i + 1}")(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f"branched_layer_{i + 2}")(hook)

        return hook


def get_padding(kernel_size):
    """
    Calculate padding values for convolutional layers.

    Args:
        kernel_size (int): Size of the convolutional kernel.

    Returns:
        list: Padding values for left and right sides of the kernel.
    """
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


class BassetBranched(L.LightningModule):
    """BassetBranched model from SJ Gosai et al., 2023;
    see <https://pmc.ncbi.nlm.nih.gov/articles/PMC10441439/>
    and original git-code: https://github.com/sjgosai/boda2/blob/main/boda/model/basset.py
    """

    ######################
    # Model construction #
    ######################

    def __init__(
        self,
        input_len=600,
        conv1_channels=300,
        conv1_kernel_size=19,
        conv2_channels=200,
        conv2_kernel_size=11,
        conv3_channels=200,
        conv3_kernel_size=7,
        n_linear_layers=1,
        linear_channels=1000,
        linear_activation="ReLU",
        linear_dropout_p=0.11625456877954289,
        n_branched_layers=3,
        branched_channels=140,
        branched_activation="ReLU",
        branched_dropout_p=0.5757068086404574,
        n_outputs=3,
        use_batch_norm=True,
        use_weight_norm=False,
        loss_criterion="L1KLmixed",
        loss_args={},
    ):
        super().__init__()

        self.input_len = input_len

        self.conv1_channels = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)

        self.conv2_channels = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        self.conv3_channels = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)

        self.n_linear_layers = n_linear_layers
        self.linear_channels = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p = linear_dropout_p

        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p = branched_dropout_p

        self.n_outputs = n_outputs

        self.loss_criterion = loss_criterion
        self.loss_args = loss_args

        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.pad1 = nn.ConstantPad1d(self.conv1_pad, 0.0)
        self.conv1 = Conv1dNorm(
            4,
            self.conv1_channels,
            self.conv1_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad2 = nn.ConstantPad1d(self.conv2_pad, 0.0)
        self.conv2 = Conv1dNorm(
            self.conv1_channels,
            self.conv2_channels,
            self.conv2_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )
        self.pad3 = nn.ConstantPad1d(self.conv3_pad, 0.0)
        self.conv3 = Conv1dNorm(
            self.conv2_channels,
            self.conv3_channels,
            self.conv3_kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=self.use_batch_norm,
            weight_norm=self.use_weight_norm,
        )

        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        next_in_channels = self.conv3_channels * self.get_flatten_factor(self.input_len)

        for i in range(self.n_linear_layers):
            setattr(
                self,
                f"linear{i + 1}",
                LinearNorm(
                    next_in_channels,
                    self.linear_channels,
                    bias=True,
                    batch_norm=self.use_batch_norm,
                    weight_norm=self.use_weight_norm,
                ),
            )
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(
            next_in_channels,
            self.branched_channels,
            self.branched_channels,
            self.n_outputs,
            self.n_branched_layers,
            self.branched_activation,
            self.branched_dropout_p,
        )

        self.output = GroupedLinear(self.branched_channels, 1, self.n_outputs)

        self.nonlin = getattr(nn, self.linear_activation)()

        self.dropout = nn.Dropout(p=self.linear_dropout_p)

    def get_flatten_factor(self, input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0

        return (hook + 2) // 4

    ######################
    # Model computations #
    ######################

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        return hook

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(self.nonlin(getattr(self, f"linear{i + 1}")(hook)))
        hook = self.branched(hook)
        return hook

    def classify(self, x):
        output = self.output(x)
        return output

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        output = self.classify(decoded)
        return output


####
# PARM MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ConvBlock(dim, dim_out = None, kernel_size = 1):
    return nn.Sequential(
        nn.BatchNorm1d(dim),
        GELU(),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = kernel_size // 2)
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        #(n p ) are length of sequence
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias = False)
    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)
        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        attn = logits.softmax(dim = -1)
        return (x * attn).sum(dim = -1)


class ResNet_Attentionpool(nn.Module):

    def __init__(self, n_block, filter_size=125, weight_file=None, 
                cell_line=False,
                type_loss='poisson', validation=False, index_interested_output=False, maxglobalpool=True,
                vocab=4, use_AttentionPool=True):
        super(ResNet_Attentionpool, self).__init__()

        self.type_loss = type_loss
        if type_loss == 'heteroscedastic': self.heteroscedastic = True
        else: self.heteroscedastic = False
        self.index_interested_output = index_interested_output
        self.validation = validation
        self.maxglobalpool = maxglobalpool
        self.vocab = vocab  # N nucleotides

        kernel_size = 7
        stem_kernel_size = 7
        
        if cell_line:
            output_nodes = len(cell_line.split('__'))
        
        else:
            output_nodes = 1

        self.n_blocks = n_block

        ##################
        # create stem
        self.stem = nn.Sequential(
                    nn.Conv1d(vocab, filter_size, stem_kernel_size, padding = "same"),
                    Residual(ConvBlock(filter_size)),
                    AttentionPool(filter_size, pool_size = 2) if use_AttentionPool else nn.MaxPool1d(2) )


        # create conv tower
        conv_layers = []

        initial_filter_size = filter_size
        prev_filter_size = filter_size
        for block in range(self.n_blocks):
            
            conv_layers.append(nn.Sequential(
                ConvBlock(prev_filter_size, filter_size, kernel_size = kernel_size),
                Residual(ConvBlock(filter_size, filter_size, kernel_size = 1)),
                AttentionPool(filter_size, pool_size = 2) if use_AttentionPool else nn.MaxPool1d(2)
            ))
            
            prev_filter_size = filter_size

        self.conv_tower = nn.Sequential(*conv_layers)
            
        self.linear1 = nn.Linear(filter_size, 1)
        if self.heteroscedastic:
            self.log_var = nn.Linear(filter_size, output_nodes)  # Log-variance output
        
        self.relu = nn.ReLU()

        #################

    def forward(self, x):

        out = self.stem(x)

        out = self.conv_tower(out)

        if self.maxglobalpool:
            #max in length
            out = torch.max(out, dim=-1).values

        out = out.view(out.size(0), -1)

        if self.heteroscedastic:
            mu = self.linear1(out)
            log_var = self.log_var(out)  # Log variance
            #return(mu)
            if self.validation: return mu
            return mu, log_var
        else:
            out = self.linear1(out)
        

        if self.type_loss == 'poisson': out = self.relu(out)
        if self.index_interested_output: out = out[:,self.index_interested_output].unsqueeze(1)


        return out.squeeze(1)

PARM = ResNet_Attentionpool

####
