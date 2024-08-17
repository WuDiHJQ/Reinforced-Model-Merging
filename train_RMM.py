import os
import torch
import argparse
import registry
from copy import deepcopy
from datetime import datetime
from engine.PPO import PPO
from engine.envs import Env_CV, Env_NLP
from engine.datasets.utils import DatasetWrapper
from engine.merge_lib import task_arithmetic, ties, dare, dare_ties
from engine.utils import DataIter, load_clip_features, reset_random, save_figure


parser = argparse.ArgumentParser(description='RLM Training')
parser.add_argument('--data_root', default='../data')
parser.add_argument('--model', default='vit_s')
parser.add_argument('--dataset', default='cub,dogs',
                    help='datasets for model merging.')
parser.add_argument('--method', default='ties',
                    help='methods for model merging.')
parser.add_argument('--data_scale', default=1.0, type=float,
                    help='data scale for model validation.')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')
    

def main():
    args = parser.parse_args()
    reset_random(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.model.startswith('vit'):
        args.task_type = 'CV'
    else:
        args.task_type = 'NLP'
    
    datasets = args.dataset.split(',')
    data_iters = []
    base_models = []
    clip_features = []
    
    for data in datasets:
        _, classes_name, train_dst,_ = registry.get_dataset(data, args.data_root)
        if args.task_type == 'CV':
            model = registry.get_model(args.model, num_classes=512, pretrained=True)
            clip_features.append(load_clip_features(classes_name, device=device))
            train_loader = torch.utils.data.DataLoader(train_dst, 
                                                       batch_size=args.batch_size, 
                                                       shuffle=True, 
                                                       num_workers=args.workers)
        elif args.task_type == 'NLP':
            (tokenizer, model) = registry.get_model(args.model)
            train_dst = DatasetWrapper(train_dst, tokenizer, device)  
            train_loader = torch.utils.data.DataLoader(train_dst, 
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       collate_fn=train_dst.collate_fn)
            
        model = model.to(device).eval()
        model.load_state_dict(torch.load(os.path.join('checkpoint', '%s_%s'%(data, args.model), 'best.pth')))
        data_iters.append(DataIter(train_loader))
        base_models.append(model)
            
    ################################### Merging ##########################################
    
    base_ckpt = [model.state_dict() for model in base_models]
    if args.task_type == 'CV':
        pt_ckpt = registry.get_model(args.model, num_classes=512, pretrained=True).to(device).state_dict()
    elif args.task_type == 'NLP':
        _, pt_model = registry.get_model(args.model)
        pt_ckpt = pt_model.to(device).state_dict()
    
    if args.method == 'ties':
        merged_state = [ties(base_ckpt,pt_ckpt)]
    elif args.method == 'dare':
        merged_state = [dare(base_ckpt,pt_ckpt)]
    elif args.method == 'dare_ties':
        merged_state = [dare_ties(base_ckpt,pt_ckpt), 
                        ties(base_ckpt,pt_ckpt), 
                        dare(base_ckpt,pt_ckpt)]
    else:
        raise NotImplementedError
    
    merge_models = []
    for state in merged_state:
        model = deepcopy(base_models[0])
        model.load_state_dict(state, strict=False)
        merge_models.append(model)
    
    ################################### Setting ##########################################
    
    env_name = '%s_%s_%s'%(args.model, args.dataset, args.method)
    
    data_scale = args.data_scale    # data scale for model validation
    max_training_episodes = 700     # break training loop if episode > max_training_episodes

    print_freq = 2                  # print avg reward in the interval
    log_freq = 2                    # log avg reward in the interval
    save_model_freq = 50            # save model frequency

    update_episode = 4              # update policy every n episodes
    K_epochs = 40                   # update policy for K epochs
    eps_clip = 0.2                  # clip parameter for PPO
    gamma = 0.99                    # discount factor
    dar_scale = 0.8                 # dynamic average reward
    
    lr_actor = 0.0003               # learning rate for actor network
    lr_critic = 0.001               # learning rate for critic network

    ###############################################
    if args.task_type == 'CV':
        env = Env_CV(base_models, merge_models, data_iters, clip_features, data_scale)
    elif args.task_type == 'NLP':
        env = Env_NLP(base_models, merge_models, data_iters, data_scale)
    env.seed(args.seed)
    
    # state/action space dimension
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    ################################## Logging ###########################################
    
    run_dir = "runs"
    if not os.path.exists(run_dir):
          os.makedirs(run_dir)

    run_dir = os.path.join(run_dir, env_name)
    if not os.path.exists(run_dir):
          os.makedirs(run_dir)

    ################ create new run ################
    run_num = 0
    current_num_files = next(os.walk(run_dir))[2]
    run_num = len(current_num_files)
    
    log_path = os.path.join(run_dir, 'log_%d.csv'%run_num)
    ckpt_path = os.path.join(run_dir, 'ckpt_%d.pth'%run_num)
    
    print("saving run at : " + run_dir)
    
    ################################## Training ##########################################
    
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_path,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    last_reward = 0

    # training loop
    for i_episode in range(1, max_training_episodes+1):

        state = env.reset()

        ###############################################
        # RLM Process
        # 1.select action in environment
        # 2.sample several path
        # 3.update agent
        # 4.remove old path
        ###############################################
        current_ep_reward = 0

        # a whole episode
        while True:

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            
            # Dynamic Average Reward
            # if done:
            #     if (i_episode-1) % (1/data_scale) == 0:
            #         last_reward = reward
            #     else:
            #         reward = dar_scale*(i_episode/max_training_episodes)*last_reward + \
            #                  (1 - dar_scale*(i_episode/max_training_episodes))*reward
            #         last_reward = reward
                
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # break; if the episode is over
            if done:
                break

        # update PPO agent
        if i_episode % update_episode == 0:
            ppo_agent.update()

        # log in logging file
        if i_episode % log_freq == 0:
            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if i_episode % print_freq == 0:
            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0

        # save model weights
        if i_episode % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + ckpt_path)
            ppo_agent.save(ckpt_path)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

    log_f.close()
    env.close()

    print("============================================================================================")
    # plot and save reward figure
    save_figure(env_name, run_num)
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    # print total training time
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    ######################################################################################


if __name__ == '__main__':
    main()