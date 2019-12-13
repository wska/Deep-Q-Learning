from models import *


############## LAYER COMPARISON #############
Agent1 = {"name": "16-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model   \
}

Agent2 = {"name": "16-8-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": model16to8toAction   \
}

Agent3 = {"name": "16-8-4-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": model16to8to4toAction   \
}
############################################

Agent4 = {"name": "16-8-2 (SGD)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": SGDtwoLayerModel   \
}


############ NODE COMPARISON ############
Agent5 = {"name": "16-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model   \
}

Agent6 = {"name": "32-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model32   \
}

Agent7 = {"name": "64-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.005,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model64   \
}
################################################