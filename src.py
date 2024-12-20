import torch
from torch import Tensor
from pyro.nn import PyroModule
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import numpy as np
from pyro.nn import PyroModule, PyroSample

class DeepGP(PyroModule):

    def __init__(
            self,
            dim_list = None,
            J_list = None,
            init_w = None,
            #这里不会写变量类型 搞定了
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        # if in_dim_list is None:
        #     in_dim_list = [1, 1, 1]
        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGP(in_dim_list[i],out_dim_list[i], J_list[i],init_w))
            # layer_list.append(SecondLayer(2 * J_list[i], out_dim_list[i],i))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        mu = x

        return mu
    
class SingleGPNoBias(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            init_w = None,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [FirstLayer(in_dim, 2 * J), SecondLayerNoBias(2 * J, out_dim)]#,init_w)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu
    
    def pred(self, x: Tensor) -> Tensor:
        z = self.layers[0](x)
        dmu = self.layers[1].pred(z) @ self.layers[0].pred(x)
        # dmu = self.layers[0].pred(x) @ self.layers[1].pred(z)
        return dmu

class SingleGP(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            init_w = None,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [FirstLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]#,init_w)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu

class SingleGPFix(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [SingleLayerFix(in_dim, 2 * J, out_dim)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = x#self.layers[1](x)

        return mu

class SingleLaplacianGP(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [FirstLaplacianLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu

class SingleCauchyGP(PyroModule):
    r"""
    A single random feature-based GP is equivalent to a two-layer Bayesian neural network.

    Attributes
    ----------
    layers: PyroModule
        The layers containing the FirstLayer and SecondLayer.
    """

    def __init__(
            self,
            in_dim: int = 1,
            out_dim: int = 1,
            J: int = 50,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension
        :param out_dim:
            The output dimension
        :param J:
            The number of random features
        """
        super().__init__()

        assert in_dim > 0 and out_dim > 0 and J > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = [FirstCauchyLayer(in_dim, 2 * J), SecondLayer(2 * J, out_dim)]
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        x = self.layers[0](x)
        mu = self.layers[1](x)

        return mu

class FirstLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Kernel is RBF kernel.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        self.layer.weight = PyroSample(dist.Normal(0., 1.0).expand([self.J, in_dim]).to_event(2))
        # self.layer.weight = PyroSample(dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu
    
    def pred(self, x: Tensor) -> Tensor:
        hid = self.layer(x).squeeze()
        dmu = torch.cat((torch.diag(torch.cos(hid)), -torch.diag(torch.sin(hid))), dim=0) / torch.sqrt(torch.tensor(self.J)) @ self.layer.weight
        # print(dmu.shape)
        # print(self.layer.weight.shape)
        # dmu = self.layer.weight @ torch.cat((torch.diag(torch.cos(hid)), -torch.diag(torch.sin(hid))), dim=0) / torch.sqrt(torch.tensor(self.J))
        return dmu

    # init_w = None,
    # init_b = None,
    # if init_w is None:
    #     init_w = torch.randn(out_dim, hid_dim).cuda()

class SecondLayertest(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            init_w_mean = None,
            init_w_var = None,
            init_b_mean=None,
            init_b_var=None,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()
        if init_w_mean is None:
            init_w_mean = torch.zeros(out_dim, hid_dim)
        else:
            init_w_mean = init_w_mean.reshape(out_dim, hid_dim)
        if init_w_var is None:
            init_w_var = torch.ones(out_dim, hid_dim)
        else:
            init_w_var = init_w_var.reshape(out_dim, hid_dim)
        if init_b_mean is None:
            init_b_mean = torch.zeros(out_dim, hid_dim)
        else:
            init_b_mean = init_b_mean.reshape(out_dim, hid_dim)
        if init_b_var is None:
            init_b_var = torch.ones(out_dim, hid_dim)
        else:
            init_b_var = init_b_var.reshape(out_dim, hid_dim)

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(init_w_mean, init_w_var))#torch.ones(out_dim,hid_dim)).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))
        self.layer.bias = PyroSample(dist.Normal(init_b_mean, init_b_var))
        #self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0, device='cuda')).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu

class SecondLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(1., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

        self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu
    
    def pred(self, x: Tensor) -> Tensor:
        dmu = self.layer.weight
        return dmu

class SingleLayerFix(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Kernel is RBF kernel.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            out_dim: int = 1,
            seed = 1
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        # self.J = hid_dim // 2
        # self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # torch.manual_seed(seed)
        # self.layer.weight =  torch.rand(self.J,in_dim)#PyroSample(dist.Normal(0., 1.).expand([self.J, in_dim]).to_event(2))
        # #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    # def forward(
    #         self,
    #         x: Tensor,
    # ) -> Tensor:
    #     r"""
    #     :param x: Tensor
    #         The input to the FirstLayer.
    #     :return: Tensor
    #         The output of the FirstLayer, which is $\phi(\Omega \times x)$.
    #     """
    #     hid = self.layer(x)
    #     mu = torch.cat((torch.sin(hid).cuda(), torch.cos(hid).cuda()), dim=-1) / torch.sqrt(torch.tensor(self.J).cuda())

        torch.manual_seed(seed)
        self.J = hid_dim // 2
        self.in_dim = in_dim
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=True)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight",
        #                                 dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = pyro.sample(f"{num_layer}-th bias",dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

        self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim]).to_event(1))
    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """

        hid = x @ torch.rand(self.J, self.in_dim).T
        x1 = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))
        mu = self.layer(x1)

        return mu

class SecondLayerNoBias(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The second layer refers to produce the GP output plus noises.

    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The second layer is based on nn.Linear where the weight($\Theta$) and bias($\Epsilon$) is defined
        by the PyroSample.
    """

    def __init__(
            self,
            hid_dim: int = 100,
            out_dim: int = 1,
            # num_layer: int = 1,
    ) -> None:
        """
        :param out_dim: int
            The output dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](hid_dim, out_dim, bias=False)
        # self.layer.weight = pyro.sample(f"{num_layer}-th Weight", dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim, hid_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Normal(0., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))
        # self.layer.bias = PyroSample(dist.Normal(0., torch.tensor(1.0,device = 'cuda')).expand([out_dim]).to_event(1))

    # known: phi(Omega x)
    # want: phi(Omega x) W + epsilon
    # epsilon : (2,)
    # Secondlayer:
    #     layer.weight: W;
    #     layer.bias: epsilon

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the SecondLayer.
        :return: Tensor
            The output of the SecondLayer, which is $\phi(\Omega \times x)\Theta + \Epsilon$.
        """
        mu = self.layer(x)

        return mu
    
    def pred(self, x: Tensor) -> Tensor:
        dmu = self.layer.weight
        return dmu

class FirstLaplacianLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    The kernel here is Laplacian kernel
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # self.layer.weight = pyro.sample(f"Laplacian {num_layer}-th Omega",
        #                                 dist.Cauchy(0., 1.).expand([self.J, in_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Cauchy(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu

class FirstCauchyLayer(PyroModule):
    r"""
    The random feature-based GP is equivalent to a two-layer Bayesian neural network.
    The first layer refers to generate the random feature $\phi$.
    Cauchy kernel is better on complex dataset, not linearly separable data.
    Attributes
    ----------
    J: int
        The number of random feature.
    layer: PyroModule
        The first layer is based on nn.Linear where the weight($\Omega$) is defined by the PyroSample.
    """

    def __init__(
            self,
            in_dim: int = 1,
            hid_dim: int = 100,
            # num_layer: int = 1,
    ) -> None:
        """
        :param in_dim: int
            The input dimension.
        :param hid_dim: int
            The hidden state's dimension, which is 2J. The hidden state here refers to the output of
            the first layer and the input of the second layer.
        """
        super().__init__()

        self.J = hid_dim // 2
        self.layer = PyroModule[nn.Linear](in_dim, self.J, bias=False)
        # self.layer.weight = pyro.sample(f"Laplacian {num_layer}-th Omega",
        #                                 dist.Laplace(0., 1.).expand([self.J, in_dim]).to_event(2))
        self.layer.weight = PyroSample(dist.Laplace(0., 1.).expand([self.J, in_dim]).to_event(2))
        #pyro.sample()

    # prior of Omega(5*4): location (5*4): 0.0; scale (5*4): 1.0
    #
    # posterior of Omega:  location (5*4): variational parameter; scale (5*4): variation parameter
    #
    # sample from posterior: N(location, scale)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        r"""
        :param x: Tensor
            The input to the FirstLayer.
        :return: Tensor
            The output of the FirstLayer, which is $\phi(\Omega \times x)$.
        """
        hid = self.layer(x)
        mu = torch.cat((torch.sin(hid), torch.cos(hid)), dim=-1) / torch.sqrt(torch.tensor(self.J))

        return mu


class DeepGPNoBias(PyroModule):
    def __init__(
            self,
            dim_list = None,
            J_list = None,
    ) -> None:
        super().__init__()
        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGPNoBias(in_dim_list[i], out_dim_list[i], J_list[i]))
            #还没写singleGPnobias
            # layer_list.append(SecondLayerNoBias(2 * J_list[i], out_dim_list[i],i))
        #layer_list = [FirstLayer(in_dim_list[i], 2 * J_list[i]), SecondLayer(2 * J_list[i], out_dim_list[i]) for i in range(len(in_dim_list))]
        #layer_list.append(PyroModule[nn.Linear](out_dim_list[-1], out_dim_list[-1], bias=True))
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
    
    def pred(self, x: Tensor) -> Tensor:
        # dmu = torch.ones(1)
        dmu = torch.eye(x.shape[1])
        for i in range(len(self.layers)):
            dmu = self.layers[i].pred(x) @ dmu
            # dmu = dmu @ self.layers[i].pred(x)
            x = self.layers[i](x)
        return dmu

class MtlDeepGP(PyroModule):
    def __init__(self, dim_list=[1, 1], dim1_list = [1, 1], dim2_list = [1, 1], J_list=[10],J1_list=[10], J2_list=[10]):
        super().__init__()

        # self.out_dim = dim_list[-1]
        self.GPcommon = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
        # self.model.to('cpu')

    def forward(self, x, y=None):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        z1 = self.GPcommon(x1)
        z2 = self.GPcommon(x2)
        z = 1/2 * (z1 + z2)
        self.z = z
        # z = z1
        y1 = self.GP1(z)
        y2 = self.GP2(z)
        y = torch.cat((y1, y2), dim=1)
        mu = y

        return mu
    
    def pred(self, x: Tensor) -> Tensor:
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        z1 = self.GPcommon(x1)
        z2 = self.GPcommon(x2)
        z = 1/2 * (z1 + z2)
        # z = z1
        y1 = self.GP1(z)
        y2 = self.GP2(z)
        y = torch.cat((y1, y2), dim=1)
        dy1dx1 = self.GP1.pred(z) * 1/2 @ self.GPcommon.pred(x1)
        dy2dx1 = self.GP2.pred(z) * 1/2 @ self.GPcommon.pred(x1)
        dy1dx2 = self.GP1.pred(z) * 1/2 @ self.GPcommon.pred(x2)
        dy2dx2 = self.GP2.pred(z) * 1/2 @ self.GPcommon.pred(x2)
        # dy1dx1 = self.GPcommon.pred(x1) @ self.GP1.pred(z) * 1/2
        # dy2dx1 = self.GPcommon.pred(x1) @ self.GP2.pred(z) * 1/2
        # dy1dx2 = self.GPcommon.pred(x2) @ self.GP1.pred(z) * 1/2
        # dy2dx2 = self.GPcommon.pred(x2) @ self.GP2.pred(z) * 1/2
        dmu = [[dy1dx1, dy2dx1], [dy1dx2, dy2dx2]]
        return dmu
    


# class MtlDeepGP_classification(PyroModule):
#     def __init__(self, dim_list=[1, 1], dim1_list = [1, 3], dim2_list = [1, 3], J_list=[10],J1_list=[10], J2_list=[10]):
#         super().__init__()
#         self.num_classes1 = dim1_list[-1]
#         self.num_classes2 = dim2_list[-1]
#         # self.out_dim = dim_list[-1]
#         self.GPcommon = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
#         self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
#         self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
#         # self.model.to('cpu')

#     def forward(self, x, y=None):
#         x1 = x[:,0:1]
#         x2 = x[:,1:2]
#         z1 = self.GPcommon(x1)
#         z2 = self.GPcommon(x2)
#         z = 1/2 * (z1 + z2)
#         self.z = z
#         # z = z1
#         y1 = self.GP1(z1)
#         y2 = self.GP2(z2)
#         y11 = torch.softmax(y1, dim=1)
#         y22 = torch.softmax(y2, dim=1)
#         y = torch.cat((y11, y22), dim=1)
#         mu = y

#         return mu


class MtlDeepGP_classification(PyroModule):
    def __init__(self, dim_list=[1, 1], dim1_list = [1, 3], dim2_list = [1, 3], J_list=[10],J1_list=[10], J2_list=[10]):
        super().__init__()
        self.num_classes1 = dim1_list[-1]
        self.num_classes2 = dim2_list[-1]
        # self.out_dim = dim_list[-1]
        self.GPcommon1 = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.GPcommon2 = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
        # self.model.to('cpu')

    def forward(self, x, y=None):
        x1 = x[:,0:1]
        x2 = x[:,1:2]
        z1 = self.GPcommon1(x1)
        z2 = self.GPcommon2(x2)
        z = 1/2 * (z1 + z2)
        self.z = z
        # z = z1
        y1 = self.GP1(z)
        y2 = self.GP2(z)
        y11 = torch.softmax(y1, dim=1)
        y22 = torch.softmax(y2, dim=1)
        y = torch.cat((y11, y22), dim=1)
        mu = y

        return mu
    


##########################################################################################
# PyroSample(dist.Normal(1., torch.tensor(1.0)).expand([out_dim, hid_dim]).to_event(2))
class Deconfounder_z2x(PyroModule): 
    def __init__(self, dim_list=[1, 1], J_list=[10], shared_z=None):
        super().__init__()
        self.GP = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.confounder = shared_z
        self.out_dim = dim_list[-1]

    def forward(self, x=None):
        z = self.confounder
        x = self.GP(z)
        mu = x
        return mu

        # scale = pyro.sample("sigma", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.out_dim) 
        # # Sampling model
        # with pyro.plate("data", x.shape[0]): 
        #     obs = pyro.sample("obs", dist.MultivariateNormal(mu.cpu(), torch.diag(scale * scale).cpu()), obs=x)
        # return obs


class Deconfounder_zx2y(PyroModule): 
    def __init__(self, dim_list=[1, 1], J_list=[10], shared_z=None):
        super().__init__()
        self.GP = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.confounder = shared_z
        self.out_dim = dim_list[-1]

    def forward(self, x, y=None):
        z = self.confounder
        v = torch.cat((z, x), dim=-1)
        y = self.GP(v)
        mu = y
        return mu

        # scale = pyro.sample("sigma", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.out_dim) 
        # # Sampling model
        # with pyro.plate("data", x.shape[0]): 
        #     obs = pyro.sample("obs", dist.MultivariateNormal(mu.cpu(), torch.diag(scale * scale).cpu()), obs=y)
        # return obs
    

    
####################################################################################
class Deconfounder_z2x_v2(PyroModule): 
    def __init__(self, dim_list=[1, 1], J_list=[10], shared_z=None):
        super().__init__()
        # self.GPs = []
        # for i in range(num_classes):
        #     self.GPs.append(DeepGPNoBias(dim_list=dim_list, J_list=J_list))
        self.GP = DeepGPNoBias(dim_list=dim_list, J_list=J_list)
        self.confounder = shared_z
        self.out_dim = dim_list[-1]

    def forward(self, c, x=None):
        z = self.confounder[c.squeeze()]
        x = self.GP(z)
        mu = x
        return mu


class Deconfounder_zx2y_v2(PyroModule): 
    def __init__(self, dim_list=[1, 1], J_list=[10], shared_z=None):
        super().__init__()
        n_class = shared_z.shape[0]
        self.GP = DeepGPNoBias_c(dim_list=dim_list, J_list=J_list, n_class=n_class)
        # self.GPs = []
        # for i in range(n_class):
        #     self.GPs.append(DeepGPNoBias(dim_list=dim_list, J_list=J_list))
        self.confounder = shared_z
        self.out_dim = dim_list[-1]

    def forward(self, x, c, y=None):
        z = self.confounder[c.squeeze()]
        # print(c.shape)
        # print(z.shape)
        v = torch.cat((z, x), dim=-1)
        y = self.GP(v, c)
        mu = y
        return mu
    


class DeepGPNoBias_c(PyroModule):
    def __init__(
            self,
            dim_list = None,
            J_list = None,
            n_class = 1
    ) -> None:
        super().__init__()
        # for i in range(len(dim_list)-1):
        #     dim_list[i] = dim_list[i] * n_class

        in_dim_list = dim_list[:-1]
        out_dim_list = dim_list[1:]

        in_dim_list[0] = in_dim_list[0] * n_class
        self.n_class = n_class

        assert min(in_dim_list) > 0 and min(out_dim_list) > 0 and min(J_list) > 0  # make sure the dimensions are valid

        # Define the PyroModule layer list
        layer_list = []
        for i in range(len(in_dim_list)):
            layer_list.append(SingleGPNoBias(in_dim_list[i], out_dim_list[i], J_list[i]))
   
        print(layer_list)
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

    def forward(
            self,
            x: Tensor,
            c
    ) -> Tensor:
        """
        :param x: Tensor
            The input into the Single GP
        :return:
            The output of the Single GP
        """
        xx = x.unsqueeze(1).repeat(1, self.n_class, 1) * 0
        xx[:, c.squeeze(), :] = x
        xx = xx.reshape(x.shape[0], x.shape[1] * self.n_class)
        for i in range(len(self.layers)):
            xx = self.layers[i](xx)
        return xx
    
    def pred(self, x: Tensor, c) -> Tensor:
        dmu = torch.ones(1)
        xx = x.unsqueeze(1).repeat(1, self.n_class, 1) * 0
        xx[:, c.squeeze(), :] = x
        xx = xx.reshape(x.shape[0], x.shape[1] * self.n_class)
        for i in range(len(self.layers)):
            dmu = self.layers[i].pred(xx) @ dmu
            xx = self.layers[i](xx)
        return dmu
    

    
####################################################################################
class Model_Nov17(PyroModule):
    def __init__(self, dim_list=[1, 1], dim1_list = [1, 3], dim2_list = [1, 3], J_list=[10],J1_list=[10], J2_list=[10]):
        super().__init__()
        self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
        self.GP3 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)

        self.GP31 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP32 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)

    def forward(self, x1, x2, x3, y1=None, y2=None):
        mu1 = self.GP1(x1)
        mu2 = self.GP2(x2)

        z = self.GP3(x3)
        mu31 = self.GP31(z)
        mu32 = self.GP32(z)
        
        scale1 = pyro.sample("sigma1", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y1_dim) 
        scale2 = pyro.sample("sigma3", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y2_dim)
        scale31 = pyro.sample("sigma21", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y1_dim)
        scale32 = pyro.sample("sigma22", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y2_dim) 
        # Sampling model
        with pyro.plate("data", x1.shape[0]): 
            obs1 = pyro.sample("obs1", dist.MultivariateNormal(mu1.cpu(), torch.diag(scale1 * scale1).cpu()), obs=y1)
            obs2 = pyro.sample("obs2", dist.MultivariateNormal(mu2.cpu(), torch.diag(scale2 * scale2).cpu()), obs=y2)
            obs31 = pyro.sample("obs31", dist.MultivariateNormal(mu31.cpu(), torch.diag(scale31 * scale31).cpu()), obs=y1)
            obs32 = pyro.sample("obs32", dist.MultivariateNormal(mu32.cpu(), torch.diag(scale32 * scale32).cpu()), obs=y2)
        return obs1, obs2, obs31, obs32
    


####################################################################################
class Model_Nov18(PyroModule):
    def __init__(self, dim1_list=[1, 1], dim2_list = [1, 1], dim3_list = [1, 1], dim31_list=[2, 1], dim32_list=[2, 1],  J1_list=[10], J2_list=[10], J3_list=[10],  J31_list=[10], J32_list=[10]):
        super().__init__()
        self.GP1 = DeepGPNoBias(dim_list=dim1_list, J_list=J1_list)
        self.GP2 = DeepGPNoBias(dim_list=dim2_list, J_list=J2_list)
        self.GP3 = DeepGPNoBias(dim_list=dim3_list, J_list=J3_list)

        self.GP31 = DeepGPNoBias(dim_list=dim31_list, J_list=J31_list)
        self.GP32 = DeepGPNoBias(dim_list=dim32_list, J_list=J32_list)

    def forward(self, x1, x2, x3, y1=None, y2=None):
        xx1 = self.GP1(x1)
        xx2 = self.GP2(x2)
        
        xx3 = self.GP3(x3)
        z1 = torch.cat((xx3, xx1), dim=-1)
        mu1 = self.GP31(z1)
        z2 = torch.cat((xx3, xx2), dim=-1)
        mu2 = self.GP32(z2)
        
        scale1 = pyro.sample("sigma1", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y1_dim)
        scale2 = pyro.sample("sigma2", dist.Gamma(torch.tensor(0.5, device='cpu'), torch.tensor(1.0, device='cpu'))).expand(self.y2_dim) 
        # Sampling model
        with pyro.plate("data", x1.shape[0]): 
            obs1 = pyro.sample("obs1", dist.MultivariateNormal(mu1.cpu(), torch.diag(scale1 * scale1).cpu()), obs=y1)
            obs2 = pyro.sample("obs2", dist.MultivariateNormal(mu2.cpu(), torch.diag(scale2 * scale2).cpu()), obs=y2)
        return obs1, obs2