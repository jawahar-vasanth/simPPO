#!/usr/bin/env python
import gym
import sys
import torch
import time
import os.path as osp
import os


from arguments import get_args
from ppo import PPO
from network import FeedForwardNN
from eval_policy import eval_policy
from eval_progress import eval_progress

import lidarCarV0


def train(env, hyperparameters, actor_model, critic_model, datapath, exp_name):
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(datapath +"/" + actor_model))
		model.critic.load_state_dict(torch.load(datapath +"/" + critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': 
		print(f"Error: Specify both actor/critic models or none at all to avoid accidental override")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)
		ymd_time = time.strftime("%m-%d-%H-%M_")
		relpath = ''.join([ymd_time, exp_name])
		datapath = osp.join(datapath, relpath)
		if not os.path.exists(datapath): os.makedirs(datapath)
	model.learn(total_timesteps=10_000_000, logpath = datapath)

def test(env, datapath, actor_model, mode):
	print(f"Testing {actor_model}", flush=True)

	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
	entries = os.listdir(datapath)
	entries = [int(x) for x in entries]
	entries.sort()

	if mode == 'test':
		policy = FeedForwardNN(obs_dim, act_dim)
		actor_model = datapath + "/" + str(entries[-1]) + "/" + actor_model
		policy.load_state_dict(torch.load(actor_model))
		print("Iteration "+ str(entries[-1]))
		eval_policy(policy=policy, env=env, render=True)
	
	if mode == 'progress':
		eval_progress(list_dir = entries, file_dir = datapath, actor_model = actor_model,\
			 obs_dim = obs_dim, act_dim = act_dim, env = env, render = True)
		'''
		for filename in entries:
			actor_model = datapath + str(filename) + "/" + actor_model
			policy = FeedForwardNN(obs_dim, act_dim)
			eval_progress(policy=policy, env=env, render=True)
		'''

def main(args):
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'save_freq': 10,
				'render_every_i': 10,
				'interm_save': True
			  }
	# env_name = 'LunarLanderContinuous-v2'
	env_name = 'Lidarcar-v0'
	env = gym.make(env_name)

	if args.exp_name == '':
    		args.exp_name = env_name

	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model,\
			critic_model=args.critic_model, datapath = args.datapath, exp_name = args.exp_name)
	else:
		test(env=env, datapath = args.datapath, actor_model = args.actor_model, mode = args.mode)

if __name__ == '__main__':
	args = get_args()
	main(args)
