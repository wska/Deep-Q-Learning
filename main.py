import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import plot_model
from models import *
from agents import *

EPISODES = 1000 #Maximum number of episodes

#DQN Agent for the Cartpole
#Q function approximation with NN, experience replay, and target network
class DQNAgent:
    #Constructor for the agent (invoked when DQN is first called in main)
    def __init__(self, state_size, action_size, discount_factor=0.95, learning_rate=0.005,\
        target_update_frequency=1, memory_size=1000, regularization = 0.000, epsilonDecay=1.00, model=default_model):

        self.check_solve = False	#If True, stop if you satisfy solution confition
        self.render = False        #If you want to see Cartpole learning, then change to True

        #Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

       # Modify here

        #Set hyper parameters for the DQN. Do not adjust those labeled as Fixed.
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.target_update_frequency = target_update_frequency
        self.memory_size = memory_size
        self.regularization = regularization
        self.epsilonDecay = epsilonDecay


        # Fixed parameters(DO NOT TOUCH)
        self.train_start = 1000 #Fixed
        self.epsilon = 0.02 #Fixed
        self.batch_size = 32 #Fixed
        
        #Number of test states for Q value plots
        self.test_state_no = 10000

        #Create memory buffer using deque
        self.memory = deque(maxlen=self.memory_size)

        #Create main network and target network (using build_model defined below)
        self.model = model(self)
        self.target_model = model(self)

        #plot_model(self.model, to_file='model.png', show_shapes=True)

        #Initialize target network
        self.update_target_model()



    #After some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    #Get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
        return action



    #Save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #Add sample to the end of the list

    #Sample <s,a,r,s'> from replay memory
    def train_model(self):
        if len(self.memory) < self.train_start: #Do not train if not enough memory
            return
        batch_size = min(self.batch_size, len(self.memory)) #Train on at most as many samples as you have in memory
        mini_batch = random.sample(self.memory, batch_size) #Uniformly sample the memory buffer
        #Preallocate network and target network input matrices.
        update_input = np.zeros((batch_size, self.state_size)) #batch_size by state_size two-dimensional array (not matrix!)
        update_target = np.zeros((batch_size, self.state_size)) #Same as above, but used for the target network
        action, reward, done = [], [], [] #Empty arrays that will grow dynamically

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0] #Allocate s(i) to the network input array from iteration i in the batch
            action.append(mini_batch[i][1]) #Store a(i)
            reward.append(mini_batch[i][2]) #Store r(i)
            update_target[i] = mini_batch[i][3] #Allocate s'(i) for the target network array from iteration i in the batch
            done.append(mini_batch[i][4])  #Store done(i)




        state_value = self.model.predict(update_input) #Generate target values for training the inner loop network using the network model
        target_val = self.target_model.predict(update_target) #Generate the target values for training the outer loop target network

        #Q Learning: get maximum Q value at s' from target network

        #Insert your Q-learning code here
        #Tip 1: Observe that the Q-values are stored in the variable target (renamed to state_values)
        #Tip 2: What is the Q-value of the action taken at the last state of the episode?
        

        for i in range(self.batch_size):
           
            if done[i]:
                state_value[i, action[i]] = reward[i]
            else:
                state_value[i, action[i]] = reward[i] + self.discount_factor*np.amax(target_val[i])
        
        
        self.model.fit(update_input, state_value, batch_size=self.batch_size, epochs=1, verbose=0)
        self.epsilon = self.epsilon*self.epsilonDecay

        return

############################################################################################################


def evaluateAgent(*args, **kwargs):

    #For CartPole-v0, maximum episode length is 200
    env = gym.make('CartPole-v0') #Generate Cartpole-v0 environment object from the gym library
    #Get state and action sizes from the environment
    print("OBSERVATION SPACE: ", end="")
    print(env.observation_space)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #Create agent, see the DQNAgent __init__ method for details
    agent = DQNAgent(state_size=state_size, action_size=action_size, *args, **kwargs)

    #Collect test states for plotting Q values using uniform random policy
    test_states = np.zeros((agent.test_state_no, state_size))
    max_q = np.zeros((EPISODES, agent.test_state_no))
    max_q_mean = np.zeros((EPISODES,1))

    done = True
    for i in range(agent.test_state_no):
        if done:
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            test_states[i] = state
        else:
            action = random.randrange(action_size)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            test_states[i] = state
            state = next_state

    scores, episodes = [], [] #Create dynamically growing score and episode counters
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset() #Initialize/reset the environment
        state = np.reshape(state, [1, state_size]) #Reshape state so that to a 1 by state_size two-dimensional array ie. [x_1,x_2] to [[x_1,x_2]]
        #Compute Q values for plotting
        tmp = agent.model.predict(test_states)
        max_q[e][:] = np.max(tmp, axis=1)
        max_q_mean[e] = np.mean(max_q[e][:])

        while not done:
            if agent.render:
                env.render() #Show cartpole animation

            #Get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size]) #Reshape next_state similarly to state

            #Save sample <s, batch dimension to 'testnote' as fa, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            #Training step
            agent.train_model()
            score += reward #Store episodic reward
            state = next_state #Propagate state

            if done:
                #At the end of every episode, update the target network
                if e % agent.target_update_frequency == 0:
                    agent.update_target_model()
                #Plot the play time for every episode
                scores.append(score)
                episodes.append(e)

                print("episode:", e, "  score:", score," q_value:", max_q_mean[e],"  memory length:",
                      len(agent.memory))

                # if the mean of scores of last 100 episodes is bigger than 195
                # stop training
                if agent.check_solve:
                    if np.mean(scores[-min(100, len(scores)):]) >= 195:
                        print("solved after", e-100, "episodes")
                        #agent.plot_data(episodes,scores,max_q_mean[:e+1])
                        max_q_mean = max_q_mean[:e+1]
    

    return episodes, scores, max_q_mean


#Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
def plot_data(episodes, scores, max_q_mean, legends):

    assert len(episodes) == len(max_q_mean)
    pylab.figure(0)
    for i in range(len(episodes)):
        pylab.plot(episodes[i], max_q_mean[i])
    pylab.xlabel("Episodes")
    pylab.ylabel("Average Q Value")
    pylab.legend(legends)
    pylab.title("Average Q-value over Episodes")
    pylab.savefig("plots/qvalues"+str(i)+".png")

    assert len(episodes) == len(scores)
    pylab.figure(1)
    for i in range(len(episodes)):
        pylab.plot(episodes[i], scores[i])
    pylab.xlabel("Episodes")
    pylab.ylabel("Score")
    pylab.legend(legends)
    pylab.title("Scores over Episodes")
    pylab.savefig("plots/scores"+str(i)+".png")



def main():
    agentEpisodes = []
    agentScores = []
    agentMax_q_means = []   
    legends = []    

    agents = [Agent1, Agent2, Agent3]



    for agent in agents:
        episode,score,max_q_mean = evaluateAgent(discount_factor=agent["discount_factor"], learning_rate=agent["learning_rate"],\
                  target_update_frequency=agent["target_update_frequency"], memory_size=agent["memory_size"],\
                  regularization = agent["regularization"], epsilonDecay=agent["epsilonDecay"], \
                  model=agent["model"])

        agentEpisodes.append(episode)
        agentScores.append(score)
        agentMax_q_means.append(max_q_mean)
        legends.append(agent["name"])


    plot_data(agentEpisodes,agentScores,agentMax_q_means, legends)




if __name__ == "__main__":
    main()