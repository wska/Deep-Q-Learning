from models import *


############## LAYER COMPARISON #############
Agent1 = {"name": "16-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": default_model   \
}

Agent2 = {"name": "16-16-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": model16to16toAction   \
}

Agent3 = {"name": "16-16-16-2 (ADAM)", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": model16to16to16toAction   \
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



Agent8 = {"name": "64-32-2 (ADAM), ED=1", \
          "discount_factor": 0.99,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": model64to32toAction   \
}

Agent9 = {"name": "64-32-2 (ADAM), ED=0.995", \
          "discount_factor": 0.99,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 0.995,   \
          "model": model64to32toAction   \
}

Agent10 = {"name": "64-32-2 (ADAM), ED=0.99", \
          "discount_factor": 0.99,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 0.99,   \
          "model": model64to32toAction   \
}


################################################


# NEW DEFAULT AGENT #
Agent11 = {"name": "32-32-2 (ADAM), LR=0.0001, MZ=1000", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.0001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.0,   \
          "model": updated_default   \
}

Agent12 = {"name": "32-32-2 (ADAM), LR=0.0001, MZ=2000", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.0001,  \
          "target_update_frequency": 1,    \
          "memory_size": 2000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.0,   \
          "model": updated_default   \
}

Agent13 = {"name": "32-32-2 (ADAM), LR=0.0001, MZ=5000", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.0001,  \
          "target_update_frequency": 1,    \
          "memory_size": 5000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.0,   \
          "model": updated_default   \
}

################################################

#Simple models
Agent14 = {"name": "16-16-16-2 (ADAM), lr=0.001, MZ=1000", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 1000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": simple_model   \
}

Agent15 = {"name": "16-16-16-2 (ADAM), lr=0.001, MZ=2500", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 2500,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": simple_model   \
}

Agent16 = {"name": "16-16-16-2 (ADAM), lr=0.001, MZ=5000", \
          "discount_factor": 0.95,    \
          "learning_rate": 0.001,  \
          "target_update_frequency": 1,    \
          "memory_size": 5000,    \
          "regularization": 0.000,     \
          "epsilonDecay": 1.00,   \
          "model": simple_model   \
}