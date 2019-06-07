# https://github.com/kayuksel/forex-rl-challenge
root = '/home/kamer/Videos/fx/'
from past.builtins import execfile
execfile('shared.py')

# Constants defining the neural network and multiprocessing
No_Features = 512
No_Channels = len(assets)
# No. processes, reduce this if it doesnt fit to your GPU!!!
No_Proccess = 1
epochs = 100
# Transaction cost that is utilized for commission expenses
cost = 0.0004
# Regularization by also predicting the self portolio return
regularization = 0.0
l2 = torch.nn.MSELoss()

# Go through assigned batches for each process to calculate
# the reward that occurs from agent's portfolio decisions
'''
Action is a vector which consists of the distribution of portolio weights to the assets, 
and the leverage up to 2x. Since that we have a long-short portfolio, the absolute sum of
the portfolio weights should sum to one (If it was a long-only portfolio, it would have
only positive values which would sum to one). That's why we first give the model output 
to the tanh and then normalize the resulting vector with the sum of its absolute values.
'''
def calculate_reward(model, loader, index, risk = 1.0, skip = None):
    epoch_weights = []
    last_action = torch.ones(No_Channels).cuda()
    last_action /= float(No_Channels)
    total_reward, pos_reward, total_mse = 0.0, 0.0, 0.0
    pos_count = 0

    for i, (features, rewards) in enumerate(loader):
        if skip is not None and skip[i]: continue
        features = features.view(-1).cuda(non_blocking=True)
        rewards = rewards.float().cuda(non_blocking=True)
        # Feed the last action back to the model as an input
        state = torch.cat([features, last_action])
        # Get a new action from the model given current state
        action = model(state)
        # Tanh activation is utilized for long/short portfolio
        weights = torch.tanh(action[:-2])
        # Up to 2x leverage is allowed for each action (position)
        certain = 0.5 + torch.sigmoid(action[-2]) / 2.0
        # Absolute portfolio value should sum to one x leverage
        weights = weights / (weights.abs().sum() * certain)
        # Calculate MSE from predicted return of the portfolio
        total_mse += l2(action[-1], (weights * rewards).sum())
        # Calculate the transaction cost due to portfolio change
        reward = (weights - last_action).abs().sum() * cost
        # Calculate portfolio return relative to the market itself
        reward -= (weights * rewards).sum() #- rewards.abs().mean()
        # Calculate portfolio return relative to the last weights
        reward += (last_action * rewards).sum()
        # Save the current action to employ it for the next step
        last_action = weights
        # Save the action history to measure and plot afterwards
        epoch_weights.append(weights.detach().cpu().numpy())
        # Do not sum the reward concluded from the first action
        if len(epoch_weights) == 1: continue
        # Future-work: risk-sensitive rl using exponential utility
        total_reward = total_reward + (reward.abs() if risk else reward)
        if reward > 0:
            pos_reward = pos_reward + reward**risk
            pos_count += 1
        torch.cuda.empty_cache()
    # Calculate the average reward for the non-skipped batches
    skipped = 0 if skip is None else sum(skip)
    total_mse = total_mse / (len(loader) - skipped)
    total_reward = total_reward / (len(loader) - skipped)
    pos_reward = pos_reward.pow(1/risk) / pos_count
    if skip is None: plot_function(epoch_weights)
    return total_reward, pos_reward, total_mse

'''
Reward at each time step, is the sum of element-wise multiplication of 
portfolio weights and asset returns (minus the transaction cost, which
is the absolute sum of portfolio weight differences times the commission.
'''
def train(model, optimizer, index, risk = 1.0):
    # Risk-sensitivity disabled if the risk factor is less than 1.0
    if risk < 1.0: risk = 0.0
    # Mark the batches that are going to be skipped in this process
    skip = [(i // (len(train_loader)//No_Proccess)) != index for i in range(len(train_loader))]
    # Calculate the average reward for the batches of this process
    total_reward, pos_reward, total_mse = calculate_reward(model, train_loader, index, risk, skip)
    train_reward = pos_reward / total_reward if risk else total_reward
    train_reward = train_reward + total_mse * regularization
    #print('train %f' % -train_reward.item())
    # Perform an optimizer on the shared model with calculated loss
    optimizer.zero_grad()
    train_reward.backward()
    #nn.utils.clip_grad_norm_(model.parameters(), max_grad)
    optimizer.step()
    torch.cuda.empty_cache()

best_reward = 1.0

if __name__ == '__main__':
    # A simple linear layer is employed as an example model for you
    model = nn.Linear(No_Features + No_Channels, No_Channels+2, bias = False).cuda().share_memory()
    '''
    model.weight.data.fill_(0)
    weights = torch.from_numpy(weights)[:No_Channels, :]
    model.weight.data[:No_Channels,:No_Features] = weights.data
    '''
    # Define the optimizer that will be utilized by all processes
    optimizer = optim.Adam(params = model.parameters(), 
                           lr = 1e-3, eps = 5e-3, weight_decay = 1e-5)
    for epoch in range(epochs):
        model.train(True)
        # For each epoch start all of the processes to update model
        processes = []
        for i in range(No_Proccess):
            p = mp.Process(target=train, args=(model, optimizer, i))
            p.start()
            processes.append(p)
        for p in processes: p.join()
        # After all of processes are done, evaluate model on test set
        model.eval()

        total_reward, pos_reward, _ = calculate_reward(model, test_loader, No_Proccess+1)
        test_reward = pos_reward / total_reward

        if test_reward > best_reward: continue
        best_reward = test_reward

        lin_weights = model.weight.data.detach().cpu().numpy()
        with open(root+'models/weights.pkl', 'wb') as f:
            cPickle.dump(lin_weights, f)

        # Save best model for participating into the competition
        torch.save(model.state_dict(), root+'models/model.pt')
