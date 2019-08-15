# https://github.com/kayuksel/forex-rl-challenge
root = '/home/kamer/Videos/fx/'
from past.builtins import execfile
execfile('shared.py')

# Constants defining the neural network and multiprocessing
No_Channels = len(assets)
No_Features = 512 + No_Channels
# No. processes, reduce this if it doesnt fit to your GPU!!!
No_Proccess = 1
epochs = 50

# Transaction cost that is utilized for commission expenses
cost = 0.0004
# Function for calculating risk-measures and plotting results
STEP_SIZE = 0.01

# Go through assigned batches for each process to calculate
# the reward that occurs from agent's portfolio decisions
def calculate_reward(model, loader, index, risk = True, policy = None, plot = False):
    epoch_weights = []
    last_action = torch.ones(No_Channels).to('cpu')
    last_action /= float(No_Channels)
    total_reward, pos_reward = 0.0, 0.0
    pos_count = 0

    for i, (features, rewards) in enumerate(loader):
        features = features.view(-1).to('cpu')
        rewards = rewards.float().to('cpu')
        # Feed the last action back to the model as an input
        state = torch.cat([features, last_action]).double()
        # Get a new action from the model given current state
        action = model.forward(state, policy) if torch.is_tensor(policy) else model.forward(state)
        # Tanh activation is utilized for long/short portfolio
        weights = torch.tanh(action[:-1])
        # Up to 2x leverage is allowed for each action (position)
        certain = 0.5 + torch.sigmoid(action[-1]) / 2.0
        # Absolute portfolio value should sum to one x leverage
        weights = weights / (weights.abs().sum() * certain)
        # Calculate the transaction cost due to portfolio change
        reward = (weights.double() - last_action.double()).abs().sum() * cost
        # Calculate portfolio return relative to the market itself
        reward -= (weights.double() * rewards.double()).sum() #- rewards.abs().mean()
        # Calculate portfolio return relative to the last weights
        reward += (last_action.double() * rewards.double()).sum()
        # Save the current action to employ it for the next step
        last_action = weights.float()
        # Save the action history to measure and plot afterwards
        epoch_weights.append(weights.detach().cpu().numpy())
        if len(epoch_weights) == 1: continue
        # Future-work: risk-sensitive rl using exponential utility
        total_reward = total_reward + (reward.abs() if risk else reward)
        if reward > 0:
            pos_reward = pos_reward + reward
            pos_count += 1
    total_reward = total_reward / len(loader)
    pos_reward = pos_reward / pos_count
    return total_reward, pos_reward, epoch_weights

#worker for multithreading
def worker(q_policies, q_results, calc_reward_args):
    while True:
        data = q_policies.get()
        if data is None:
            #print('Worker Exiting')
            q_policies.task_done()
            break
        policy_d = data[0]
        delta_for_policy = data[1]
        #print('Worker recieved work - Starting Calculation')
        total_reward, pos_reward, _ = calculate_reward(calc_reward_args[0],calc_reward_args[1],
                                                    calc_reward_args[2], calc_reward_args[3],
                                                    policy_d)
        train_reward = total_reward / pos_reward if calc_reward_args[3] else total_reward
        
        q_results.put([train_reward.item(), delta_for_policy, data[2]])
        q_policies.task_done()

#biggest changes to the train algo
def train(model, index, risk = True):
    # Calculate the average reward for the batches of this process
    n_process = 24
    step_size = STEP_SIZE
    num_best_deltas = n_process // 2
    deltas = model.genDeltas()
    policy = model.weights
    if (len(deltas) < num_best_deltas): raise ValueError('Number of usable deltas less than total amount of deltas')
    q_Deltas_Policies = JoinableQueue()
    q_Rewards = Queue(maxsize=len(deltas)*2)
    
    procs = []
    #rollout policies with noise and calculate reward
    for i in range(n_process):
        calculate_reward_args = [model, train_loader, index, risk]
        p = Process(target = worker, args=(q_Deltas_Policies, q_Rewards, calculate_reward_args,))
        p.start()
        procs.append(p)
    
    for idx, delta in enumerate(deltas):
        polPlus = policy + delta
        polMinus = policy - delta
        q_Deltas_Policies.put([polPlus,idx, 0])
        q_Deltas_Policies.put([polMinus,idx, 1])

    for i in range(n_process): q_Deltas_Policies.put(None)
    q_Deltas_Policies.join()

    r = []
    for d in range(len(deltas)*2):
        [rew, d, sign] = q_Rewards.get()
        r.append([rew,d, sign])
    
    deltaDF = pd.DataFrame(columns=['Rewards', 'Deltas', 'PlusMinus'])
    for i in range(len(r)): deltaDF.loc[i] = [r[i][x] for x in range(3)]

    rewardStd = deltaDF['Rewards'].std()
    groupedDeltas = deltaDF.groupby('Deltas', as_index=True).sum()
    delta_reward_policy = torch.zeros_like(policy)
    
    #calculate the policy update step
    for idx, row in deltaDF.groupby('Deltas').sum().iterrows():
        if idx > num_best_deltas: break
        plus = deltaDF.loc[(deltaDF['Deltas'] == idx) & (deltaDF['PlusMinus'] == 0), 'Rewards'].values
        minus = deltaDF.loc[(deltaDF['Deltas'] == idx) & (deltaDF['PlusMinus'] == 1), 'Rewards'].values
        delta_reward_policy += (plus[0]-minus[0]) * deltas[int(idx)]

    #apply the update step to the policy
    return policy + (step_size / (num_best_deltas*rewardStd)) * delta_reward_policy

class ARSModel():
    def __init__(self, numFeatures, NumChannels, preload):
        self.feats = numFeatures
        self.chnls = NumChannels
        self.weights = torch.zeros(self.feats, self.chnls+1).to('cpu')
        self.weights[:,:] = torch.Tensor(preload.T)[:,:self.chnls+1]
        self.numDeltas = 36
        self.std = 0.5
        self.noiseStd = torch.tensor(self.std, dtype = torch.float64).to('cpu')

    def genDeltas(self):
        deltas = torch.randn(self.numDeltas, self.feats, self.chnls+1).to('cpu')
        return deltas * self.noiseStd.float()

    def forward(self, state, policy = None):
        if torch.is_tensor(policy): return torch.matmul(state.double(), policy.double())
        return torch.matmul(state.double(), self.weights.double())

best_reward = 1.0

if __name__ == '__main__':

    with open(root+'models/weights.pkl', 'rb') as f: weights = cPickle.load(f)
    model = ARSModel(No_Features, No_Channels, weights)
    #_, _, epoch_weights = calculate_reward(model, test_loader, No_Proccess+1, risk=True)
    #plot_function(epoch_weights)

    for epoch in range(epochs):
        start_time = time.time()
        n_policy = train(model, 0)
        total_reward, pos_reward, epoch_weights = calculate_reward(
            model, test_loader, No_Proccess+1, risk=True, policy=n_policy)
        test_reward = pos_reward / total_reward
        model.weights = n_policy

        end_time = time.time()
        threadtime = end_time-start_time
        ttime = threadtime * (epochs-1-epoch)
        h = int(ttime/(60*60))
        m = int((ttime - h*60*60)/60)
        s = int((ttime-h*60*60-m*60))
        print('Epoch {} took {} seconds. Time remaining {}h:{}m:{}s'.format(epoch+1, int(threadtime), h,m,s))
        STEP_SIZE *= 0.995
        print('Step size decreased to {}'.format(STEP_SIZE))

        if test_reward > best_reward: continue
        best_reward = test_reward

        weight_names = root+'models/policy_weights'
        np.save(weight_names, n_policy.numpy()) 
        plot_function(epoch_weights)
