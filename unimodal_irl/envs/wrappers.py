
import numpy as np
import gym
from gym import spaces


def d_ConvenienceEnv(SuperEnv):
    """Wrap an Environment so it has some convenience properties and methods"""
    
    class ConvenienceEnv(object):
        """Convenience wrapper for an environment"""
        
        def __init__(self, *args, **kwargs):
            self._self = SuperEnv(*args, **kwargs)
            
            self.states = np.arange(self._self.observation_space.n)
            self.actions = np.arange(self._self.action_space.n)
            
        def __getattribute__(self, s):
            try:
                x = super(ConvenienceEnv, self).__getattribute__(s)
            except AttributeError:
                pass
            else:
                return x
            x = self._self.__getattribute__(s)
            return x
    
    return ConvenienceEnv

