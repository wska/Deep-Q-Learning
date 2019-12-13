from models import *

Agent1 = {"name": "Agent1", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model   \
}

Agent2 = {"name": "Agent2", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": twoLayerModel   \
}

Agent3 = {"name": "Agent3", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 3,    \
          "memory_size": 1500,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": SGDtwoLayerModel   \
}