from abc import ABC,abstractmethod
import torch

__CONDITION_METHODS__ = {}

def register_condition_method(name):
    def wrapper(cls):
        if __CONDITION_METHODS__.get(name,None):
            raise NameError(f"Condition method {name} already exists")
        __CONDITION_METHODS__[name] = cls
        return cls
    return wrapper

def get_condition_method(name:str,operator, noise, **kwargs):
    if not __CONDITION_METHODS__.get(name,None):
        raise NameError(f"Condition method {name} does not exist")
    return __CONDITION_METHODS__[name](operator, noise, **kwargs)


class ConditionMethod(ABC):
    def __init__(self,operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

        
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            #Y-Ax
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            #||Y-Ax||^2
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_condition_method(name='dps')
class PosteriorSampling(ConditionMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_next=x_t -norm_grad * self.scale
        return x_next, norm