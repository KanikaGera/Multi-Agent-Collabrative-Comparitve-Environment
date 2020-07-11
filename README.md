[//]: # (Image References)

[image1]: videos/tennis_trained.gif "Trained Agent"

# Table Tennis Playing 
## Collabrative and Competitive Agents

### Project Details

Project is about training the agents to collabratively play a table tennis game and at same time compete with each other to score, their best. The environemnt used is [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and the best score achieved in this solution is 0.9 .The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

#### Collabrative Training

In this project two agents collabrate with  each with other by sharing each others perspective of the state of the environment, and sharing a common replay buffer to store experience.

#### Competitive Training

Each agent is trained individually with reward gained in individual context. Each agent has its own Actor and Critic Network to be trained for its best performance.

### Getting Started
1. Install anaconda using installer at https://www.anaconda.com/products/individual according to your Operating System.

2. Create (and activate) a new environment with Python 3.6.
    ```bash
    conda create --name drl python=3.6 
    conda activate drl
    ```
    
3. Clone the repository, and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/KanikaGera/Multi-Agent-Collabrative-Comparitve-Environment.git
    cd python 
    conda install pytorch=0.4.1 cuda90 -c pytorch
    pip install .
    ```

### Structure of Repository
1. `Train_Tennis.py`  is python code to interact with unity environment and train the agent.
2. `nohup.out` is output file consisting of training logs , while training agents.

3. `Train_Tennis.ipynb` is jupyter notebook format of Train_Tennis.py for easier access of code. The output of cell calling training function has scores until 10000 episodes. Actual environment was solved in 12000 episodes.

2. `model.py` consists of structure of RL model coded in pytorch.
    
3. `ddpg_agent.py` consist of DDPG Algorithm Implementation .
    
4. `saved/solution_actor_local_`[1 or 2]`.pth`  is saved trained model with weights for local actor network for agent 1 and 2 respectively.

5. `saved/solution_actor_target_`[1 or 2]`.pth` is saved trained model with weights for target actor network for agent 1 and 2 respectively.

6. `saved/solution_critic_local_`[1 or 2]`.pth` is saved trained model with weights for local critic network for agent 1 and 2 respectively.

7. `saved/solution_critic_target_`[1 or 2]`.pth` is saved trained model with weights for target critic network for agent 1 and 2 respectively.

8. `saved/scores.list` is saved scores while training model.

9. `videos` folder consist of video clipping of trained agents playing.

### Instructions
<ol>
    <li> Install Dependies by following commands in <b>Getting Started</b> </li>
    <li> Download the environment from one of the links below.  You need only select the environment that matches your operating system:
        <ul>
            <li>Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)</li>
            <li>Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)</li>
            <li>Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)</li>
            <li>Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)</li>
        </ul>
  
   <p>
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system. 
   </p>
   
   <p>
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)
   </p>
  </li>

  <li> Place the file in the GitHub repository, in the main folder, and unzip (or decompress) the file. </li>
</ol>

#### Train the Agents on Jupyter Notebook
<ul>
<li> Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drl` environment.

   `python -m ipykernel install --user --name drl --display-name "drl"`
 </li> 
   
<li> Before running code in a notebook, change the kernel to match the `drl` environment by using the drop-down `Kernel` menu. </li>

<li> Open Train_Tennis.ipynb  </li>

<li> Run Jupyter Notebook </li>

<li> Run the cells to train the model. </li>
</ul>

#### Train the Agents in Terminal Background  
<ul>
<li> Activate drl envionment

    `conda activate drl `
</li>
    
<li> nohup is for training in background. 

    `nohup python -u Train_Tennis.py &`
</li>
    
<li> Output can be seen in real-time in <i>nohup.out</i> file.

    `tail -f nohup.out`
</li>
</ul>   '

#### Evaluate Agents
<ul>
<li> Open Evaluate_Tennis.ipynb. The first half of notebook consist of plotting of scores and average score during training.</li> 
<li> Run the cells to plot the graph and analyze the rewards achieved during training. </li>
<li> Test the trained agents , by running the cells marked for testing in the notebook. </li>
</ul>
    
### Implementation Details
Multi Agent Deep Deterministic Policy Gradient Algorithm is used to train  Report is attached to main folder for detailed anaylis.
