import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy 
import torch 
global actions
global tmp_action
import kernal


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    #前馈的结果是在全连接层输出一个值，相当于做一个动作
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon # 增加随机性的，总有小部分动作是随机产生的
        self.eps_min = eps_end # 改变epsilon, 整个实验过程，epsilon逐渐减小
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]#列表生成式
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        if self.mem_cntr %self.mem_size == 0:
            # print (action)
            print('queue is full')
            print('**********')
        # self.state_memory[index] = np.concatenate((state.agents[0].ravel(), state.compet.ravel()))
        # self.new_state_memory[index] = np.concatenate((state_.agents[0].ravel(), state_.compet.ravel()))
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        tmp_action = torch.tensor([[-41.0010, -40.2450, -40.6053, -37.7748, -40, -40]],requires_grad=True)
        if np.random.random() > self.epsilon:
            print(observation)
            try:
                state = T.tensor(np.array(observation)).to(self.Q_eval.device)#state是当前的状态,也就是observation
                actions = self.Q_eval.forward(state)#前馈依据state做动作
                tmp_action = copy.copy(actions)
                print('forward')
                # print(tmp_action)
            except:
                # actions = copy.copy(actions)
                print('copy')
                actions = copy.copy(tmp_action)
            # actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item() # 贪婪算法，只取最高的reward值的动作
        else:
            action = np.random.choice(self.action_space)
            print('random')

        return action

    def learn(self):#对batch进行操作
        if self.mem_cntr < self.batch_size:
            # 我们要以batch size为单位训练，刚开始1,2,3，时候我们不想训练，因为没有达到batch size标准，泛化性能很差
            return
        # print('learn')
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        # 从max_mem的index中选择 batch size数目的index
        batch = np.random.choice(max_mem, self.batch_size, replace=False)#从memory中随机抽取batch_size个数据
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 此处batch & batch_size都是[ 很多index ]， batch_index是重新排序后的index， 可以测试输出
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        # print("batch:",batch)
        # print("batch_index:",batch_index)
        # print(self.state_memory.shape)
        # state_batch = T.tensor([self.state_to_array(state) for state in self.state_memory[batch]]).to(self.Q_eval.device)
        # new_state_batch = T.tensor([self.state_to_array(state) for state in self.new_state_memory[batch]]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)
        # 得到目前action对应值
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]#q_eval是当前状态下的动作
        q_next = self.Q_eval.forward(new_state_batch)#q_next是下一个状态下的动作
        q_next[terminal_batch] = 0.0

        # 根据公式，q value等于目前的奖励值 + discount factor乘以之后的奖励值
        # 后续gamma设置为0.99，说明我们非常重视未来收益
        # 此处说明了deep q learning是一个典型的value-based rf 算法
        # 它唯一的policy就是选择最大值，贪婪策略
        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]#q_target是目标状态下的动作
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()#反向传播
        self.Q_eval.optimizer.step()#更新参数

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min