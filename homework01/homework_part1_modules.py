#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# **Module** is an abstract class which defines fundamental methods necessary for a training a neural network. You do not need to change anything here, just read the comments.

# In[2]:


class Module(object):
    """
    Basically, you can think of a module as of a something (black box) 
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`: 
        
        output = module.forward(input)
    
    The module should be able to perform a backward pass: to differentiate the `forward` function. 
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule. 
    
        input_grad = module.backward(input, output_grad)
    """
    def __init__ (self):
        self._output = None
        self._input_grad = None
        self.training = True
    
    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        self._output = self._compute_output(input)
        return self._output

    def backward(self, input, output_grad):
        """
        Performs a backpropagation step through the module, with respect to the given input.
        
        This includes 
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self._input_grad = self._compute_input_grad(input, output_grad)
        self._update_parameters_grad(input, output_grad)
        return self._input_grad
    

    def _compute_output(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which will be stored in the `_output` field.

        Example: in case of identity operation:
        
        output = input 
        return output
        """
        raise NotImplementedError
        

    def _compute_input_grad(self, input, output_grad):
        """
        Returns the gradient of the module with respect to its own input. 
        The shape of the returned value is always the same as the shape of `input`.
        
        Example: in case of identity operation:
        input_grad = output_grad
        return input_grad
        """
        
        raise NotImplementedError
    
    def _update_parameters_grad(self, input, output_grad):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass
    
    def zero_grad(self): 
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass
        
    def get_parameters(self):
        """
        Returns a list with its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
        
    def get_parameters_grad(self):
        """
        Returns a list with gradients with respect to its parameters. 
        If the module does not have parameters return empty list. 
        """
        return []
    
    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True
    
    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False
    
    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Module"


# # Sequential container

# **Define** a forward and backward pass procedures.

# In[3]:


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially. 
         
         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `_output`. 
    """
    
    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []
   
    def add_module(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def _compute_output(self, input):
        """
        Basic workflow of FORWARD PASS:
        
            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})   
            
            
        Just write a little loop. 
        """
        activation = input
        for idx, layer in enumerate(self.modules):
            activation = layer.forward(activation)
        return activation


    def _compute_input_grad(self, input, output_grad):
        """
        Workflow of BACKWARD PASS:
            
            g_{n-1} = module[n-1].backward(y_{n-2}, output_grad)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)   
            grad_input = module[0].backward(input, g_1)   
             
             
        !!!
                
        To each module you need to provide the input, module saw while forward pass, 
        it is used while computing gradients. 
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass) 
        and NOT `input` to this Sequential module. 
        
        !!!
        
        """

        if len(self.modules) == 1:
            return module[0].backward(input, output_grad) 
        
        for i in range(len(self.modules) - 2, -1, -1):
            output_grad = self.modules[i+1].backward(self.modules[i]._output, output_grad)
        return self.modules[0].backward(input, output_grad)
      

    def zero_grad(self): 
        for module in self.modules:
            module.zero_grad()
    
    def get_parameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.get_parameters() for x in self.modules]
    
    def get_parameters_grad(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.get_parameters_grad() for x in self.modules]
    
    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string
    
    def __getitem__(self, x):
        return self.modules.__getitem__(x)
    
    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()
    
    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


# # Layers

# ## 1. Linear transform layer
# Also known as dense layer, fully-connected layer, FC-layer, InnerProductLayer (in caffe), affine transform
# - input:   **`batch_size x n_feats1`**
# - output: **`batch_size x n_feats2`**

# In[10]:


class Linear(Module):
    """
    A module which applies a linear transformation 
    A common name is fully-connected layer, InnerProductLayer in caffe. 
    
    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()
       
        # This is a nice initialization
        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)
        
    def _compute_output(self, input):
        output = np.dot(input, self.W.T) + self.b
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = np.dot(output_grad, self.W)
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradW = np.dot(output_grad.T, input) 
        self.gradb = np.sum(output_grad, axis=0, keepdims=False)
    
    def zero_grad(self):
        self.gradW.fill(0)
        self.gradb.fill(0)
        
    def get_parameters(self):
        return [self.W, self.b]
    
    def get_parameters_grad(self):
        return [self.gradW, self.gradb]
    
    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1], s[0])
        return q


# ## 2. SoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{softmax}(x)_i = \frac{\exp x_i} {\sum_j \exp x_j}$
# 
# Recall that $\text{softmax}(x) == \text{softmax}(x - \text{const})$. It makes possible to avoid computing exp() from large argument.

# In[11]:


class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()
    
    def _compute_output(self, input):
        # start with normalization for numerical stability
        output = np.subtract(input, input.max(axis=1, keepdims=True))
        output = np.exp(input) / np.exp(input).sum(axis=1) [:,None]
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = self._output * (output_grad -(output_grad * self._output).sum(axis=1)[:,None])
        return grad_input
    
    def __repr__(self):
        return "SoftMax"


# ## 3. LogSoftMax
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# $\text{logsoftmax}(x)_i = \log\text{softmax}(x)_i = x_i - \log {\sum_j \exp x_j}$
# 
# The main goal of this layer is to be used in computation of log-likelihood loss.

# In[15]:


class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()
    
    def _compute_output(self, input):
        # start with normalization for numerical stability
        output = np.subtract(input, input.max(axis=1, keepdims=True))

        output = input - np.log(np.exp(input).sum(axis=1)[:,None])
        return output
    
    def _compute_input_grad(self, input, output_grad):
        b, n = input.shape
        grad_input = np.repeat(np.expand_dims(np.eye(n), axis=0), b, axis=0) - np.expand_dims(
            np.exp(input) / np.exp(input).sum(axis=1) [:,None],axis=-1)
        grad_input = np.einsum('BNi,Bi ->BN', grad_input, output_grad)
        return grad_input
    
    def __repr__(self):
        return "LogSoftMax"


# ## 4. Batch normalization
# One of the most significant recent ideas that impacted NNs a lot is [**Batch normalization**](http://arxiv.org/abs/1502.03167). The idea is simple, yet effective: the features should be whitened ($mean = 0$, $std = 1$) all the way through NN. This improves the convergence for deep models letting it train them for days but not weeks. **You are** to implement the first part of the layer: features normalization. The second part (`ChannelwiseScaling` layer) is implemented below.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**
# 
# The layer should work as follows. While training (`self.training == True`) it transforms input as $$y = \frac{x - \mu}  {\sqrt{\sigma + \epsilon}}$$
# where $\mu$ and $\sigma$ - mean and variance of feature values in **batch** and $\epsilon$ is just a small number for numericall stability. Also during training, layer should maintain exponential moving average values for mean and variance: 
# ```
#     self.moving_mean = self.moving_mean * alpha + batch_mean * (1 - alpha)
#     self.moving_variance = self.moving_variance * alpha + batch_variance * (1 - alpha)
# ```
# During testing (`self.training == False`) the layer normalizes input using moving_mean and moving_variance. 
# 
# Note that decomposition of batch normalization on normalization itself and channelwise scaling here is just a common **implementation** choice. In general "batch normalization" always assumes normalization + scaling.

# In[ ]:


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = 0.
        self.moving_variance = 1.

    def _compute_output(self, input): 
        mu = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        
        if self.training:
            self.moving_mean = self.moving_mean * self.alpha + (1 - self.alpha) * mu
            self.moving_variance = self.moving_variance * self.alpha + (1 - self.alpha) * var
            
            output = (input - mu) / np.sqrt(var + self.EPS)
        else:
            output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)

        return output

    def _compute_input_grad(self, input, output_grad):
        var = np.var(input, axis=0)
        grad_input = output_grad / var
        
        b = input.shape[0]
        mu = np.mean(input, axis=0)
        var = np.var(input, axis=0)
        input_mu = input - mu
        std_inv = 1. / np.sqrt(var + self.EPS)

        dvar = np.sum(output_grad * input_mu, axis=0) * (-0.5 * std_inv**3)
        dmu = np.sum(output_grad * -std_inv, axis=0) + dvar * np.mean(-2. * input_mu, axis=0)

        grad_input = (output_grad * std_inv) + (dvar * 2 * input_mu / b) + (dmu / b)
        return grad_input

    def __repr__(self):
        return "BatchNormalization"


# In[ ]:


class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)
        
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def _compute_output(self, input):
        output = input * self.gamma + self.beta
        return output
        
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * self.gamma
        return grad_input
    
    def _update_parameters_grad(self, input, output_grad):
        self.gradBeta = np.sum(output_grad, axis=0)
        self.gradGamma = np.sum(output_grad*input, axis=0)
    
    def zero_grad(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)
        
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_parameters_grad(self):
        return [self.gradGamma, self.gradBeta]
    
    def __repr__(self):
        return "ChannelwiseScaling"


# Practical notes. If BatchNormalization is placed after a linear transformation layer (including dense layer, convolutions, channelwise scaling) that implements function like `y = weight * x + bias`, than bias adding become useless and could be omitted since its effect will be discarded while batch mean subtraction. If BatchNormalization (followed by `ChannelwiseScaling`) is placed before a layer that propagates scale (including ReLU, LeakyReLU) followed by any linear transformation layer than parameter `gamma` in `ChannelwiseScaling` could be freezed since it could be absorbed into the linear transformation layer.

# ## 5. Dropout
# Implement [**dropout**](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). The idea and implementation is really simple: just multimply the input by $Bernoulli(p)$ mask. Here $p$ is probability of an element to be zeroed.
# 
# This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons.
# 
# While training (`self.training == True`) it should sample a mask on each iteration (for every batch), zero out elements and multiply elements by $1 / (1 - p)$. The latter is needed for keeping mean values of features close to mean values which will be in test mode. When testing this module should implement identity transform i.e. `output = input`.
# 
# - input:   **`batch_size x n_feats`**
# - output: **`batch_size x n_feats`**

# In[ ]:


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        
        self.p = p
        self.mask = []
        
    def _compute_output(self, input):
        if self.training:
            self.mask = (np.random.rand(*input.shape) > self.p)
            output = 1 / (1 - self.p) * self.mask * input 
        else:
            output = input
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = 1 / (1 - self.p) * self.mask * output_grad
        return grad_input
        
    def __repr__(self):
        return "Dropout"


# # Activation functions

# Here's the complete example for the **Rectified Linear Unit** non-linearity (aka **ReLU**): 

# In[ ]:


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()
    
    def _compute_output(self, input):
        output = np.maximum(input, 0)
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = np.multiply(output_grad , input > 0)
        return grad_input
    
    def __repr__(self):
        return "ReLU"


# ## 6. Leaky ReLU
# Implement [**Leaky Rectified Linear Unit**](http://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29%23Leaky_ReLUs). Expriment with slope. 

# In[ ]:


class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()
            
        self.slope = slope
        
    def _compute_output(self, input):
        output = np.maximum(input, self.slope * input)
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * (input >= 0) 
        grad_input += output_grad * (input < 0) * self.slope
        return grad_input
    
    def __repr__(self):
        return "LeakyReLU"


# ## 7. ELU
# Implement [**Exponential Linear Units**](http://arxiv.org/abs/1511.07289) activations.

# In[ ]:


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()
        
        self.alpha = alpha
        
    def _compute_output(self, input):
        output = input * (input > 0) + self.alpha * (np.exp(input) - 1) * (input <= 0)
        return output
    
    def _compute_input_grad(self, input, output_grad):
        grad_input = output_grad * (input >= 0) 
        grad_input += output_grad * (input < 0) * self.alpha * np.exp(input)
        return grad_input
    
    def __repr__(self):
        return "ELU"


# ## 8. SoftPlus
# Implement [**SoftPlus**](https://en.wikipedia.org/wiki%2FRectifier_%28neural_networks%29) activations. Look, how they look a lot like ReLU.

# In[ ]:


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()
    
    def _compute_output(self, input):
        output = np.log(1 + np.exp(input))
        return output
    
    def _compute_input_grad(self, input, output_grad):        
        grad_input = output_grad * (1 - 1 / (1 + np.exp(input)))
        return grad_input
    
    def __repr__(self):
        return "SoftPlus"


# # Criterions

# Criterions are used to score the models answers. 

# In[ ]:


class Criterion(object):
    def __init__ (self):
        self._output = None
        self._input_grad = None
        
    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function 
            associated to the criterion and return the result.
            
            For consistency this function should not be overrided,
            all the code goes in `_compute_output`.
        """
        self._output = self._compute_output(input, target)
        return self._output

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result. 

            For consistency this function should not be overrided,
            all the code goes in `_compute_input_grad`.
        """
        self._input_grad = self._compute_input_grad(input, target)
        return self._input_grad
    
    def _compute_output(self, input, target):
        """
        Function to override.
        """
        raise NotImplementedError

    def _compute_input_grad(self, input, target):
        """
        Returns gradient of input wrt output
        
        Function to override.
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want 
        to have readable description. 
        """
        return "Criterion"


# The **MSECriterion**, which is basic L2 norm usually used for regression, is implemented here for you.
# - input:   **`batch_size x n_feats`**
# - target: **`batch_size x n_feats`**
# - output: **scalar**

# In[ ]:


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()
        
    def _compute_output(self, input, target):   
        output = np.sum(np.power(input - target,2)) / input.shape[0]
        return output 
 
    def _compute_input_grad(self, input, target):
        grad  = (input - target) * 2 / input.shape[0]
        return grad

    def __repr__(self):
        return "MSECriterion"


# ## 9. Negative LogLikelihood criterion (numerically unstable)
# You task is to implement the **ClassNLLCriterion**. It should implement [multiclass log loss](http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss). Nevertheless there is a sum over `y` (target) in that formula, 
# remember that targets are one-hot encoded. This fact simplifies the computations a lot. Note, that criterions are the only places, where you divide by batch size. Also there is a small hack with adding small number to probabilities to avoid computing log(0).
# - input:   **`batch_size x n_feats`** - probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# 

# In[ ]:


class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()
        
    def _compute_output(self, input, target): 
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        b = input.shape[0]
        output = - np.sum(target * np.log(input)) / b
        return output

    def _compute_input_grad(self, input, target):
        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        
        b = input.shape[0]
        grad = - target / input_clamp / b
        return grad
    
    def __repr__(self):
        return "ClassNLLCriterionUnstable"


# ## 10. Negative LogLikelihood criterion (numerically stable)
# - input:   **`batch_size x n_feats`** - log probabilities
# - target: **`batch_size x n_feats`** - one-hot representation of ground truth
# - output: **scalar**
# 
# Task is similar to the previous one, but now the criterion input is the output of log-softmax layer. This decomposition allows us to avoid problems with computation of forward and backward of log().

# In[ ]:


class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
        
    def _compute_output(self, input, target): 
        b = input.shape[0]
        output = - np.sum(target * (input - np.log(np.exp(input).sum(axis=1) [:,None]))) / b
        return output

    def _compute_input_grad(self, input, target):
        b = input.shape[0]
        return -target / b
    
    def __repr__(self):
        return "ClassNLLCriterion"


# # Optimizers

# In[ ]:


class Optimizer(object):
    def __init__(self, network):
        self._network = network  # contains trainable paramenters and their gradients
        self._state = {}  # any information needed to save between optimizer iterations

    def step(self):
        """
        Updates network parameters
        """
        raise NotImplementedError


# ### SGD optimizer with momentum

# On each step it uses the following formulas for network parameters update:
# $$v_{t+1} = \mu * v_t + g_{t+1}$$
# $$p_{t+1} = p_t - \alpha * v_{t+1}$$
# Where $p_t$ - network parameters, $v_t$ - velocity, $\mu$ - momentum, $\alpha$ - learning rate, $g_t$ - gradients.
# 
# Check `torch.optim.SGD` documentation

# In[ ]:


class SGD(Optimizer):
    def __init__(self, network, lr, momentum=0.0):
        super(SGD, self).__init__(network)
        self._learning_rate = lr
        self._momentum = momentum
        
    def step(self):
        variables = self._network.get_parameters()
        gradients = self._network.get_parameters_grad()
        
        # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
        self._state.setdefault('accumulated_grads', {})
    
        var_index = 0 
        for current_layer_vars, current_layer_grads in zip(variables, gradients):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
                old_grad = self._state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))
                np.add(self._momentum * old_grad, current_grad, out=old_grad)
                current_var -= self._learning_rate * old_grad
                var_index += 1


# ## 11. [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer
# Formulas for optimizer:
# 
# Current step learning rate: $$\text{lr}_t = \text{learning_rate} * \frac{\sqrt{1-\beta_2^t}} {1-\beta_1^t}$$
# First moment of var: $$\mu_t = \beta_1 * \mu_{t-1} + (1 - \beta_1)*g$$ 
# Second moment of var: $$v_t = \beta_2 * v_{t-1} + (1 - \beta_2)*g*g$$
# New values of var: $$\text{variable} = \text{variable} - \text{lr}_t * \frac{\mu_t}{\sqrt{v_t} + \epsilon}$$

# In[ ]:


class Adam(Optimizer):
    def __init__(self, network, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super(Adam, self).__init__(network)
        self._learning_rate = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._epsilon = eps
        
    def step(self):
        variables = self._network.get_parameters()
        gradients = self._network.get_parameters_grad()
        
        self._state.setdefault('m', {})  # first moment vars
        self._state.setdefault('v', {})  # second moment vars
        self._state.setdefault('t', 0)   # timestamp
        self._state['t'] += 1
        t = self._state['t']
    
        var_index = 0 
        lr_t = self._learning_rate * np.sqrt(1 - self._beta2**t) / (1 - self._beta1**t)
        for current_layer_vars, current_layer_grads in zip(variables, gradients): 
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
                var_first_moment = self._state['m'].setdefault(var_index, np.zeros_like(current_grad))
                var_second_moment = self._state['v'].setdefault(var_index, np.zeros_like(current_grad))

                prev_first_moment = self._beta1 * var_first_moment
                np.add(prev_first_moment, (1 - self._beta1) * current_grad , var_first_moment)
                
                prev_second_moment = self._beta2 * var_second_moment
                np.add(prev_second_moment, (1 - self._beta2) * current_grad * current_grad, var_second_moment)
                
                np.add(current_var, -lr_t * var_first_moment / (np.sqrt(var_second_moment) + self._epsilon), 
                       out=current_var)
    
                # small checks that you've updated the state; use np.add for rewriting np.arrays values
                assert var_first_moment is self._state['m'].get(var_index)
                assert var_second_moment is self._state['v'].get(var_index)
                var_index += 1

