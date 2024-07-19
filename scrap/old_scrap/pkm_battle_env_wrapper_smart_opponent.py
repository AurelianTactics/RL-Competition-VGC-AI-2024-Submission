'''
Wraps PkmBattleEnv into something expected by most RL algorithms

to do when starting competition:
turn on random teams
have agent swap sides
'''

import gymnasium as gym
import numpy as np

from vgc.datatypes.Objects import PkmTeam
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator




class PkmBattleEnvWrapper(gym.Wrapper):

    def __init__(self, id: str):
        team0, team1 = PkmTeam(), PkmTeam()
        
        # competition rules
        # need to test and disable the set team when set team shows it works
        # self.team_generator = RandomTeamGenerator(2)
        # self.env = self._get_random_team_env(self.team_generator)

        # set team for testing
        self.env = PkmBattleEnv((team0, team1),
                   # encode Fasle for forward env
                   #encode=(agent0.requires_encode(), agent1.requires_encode()))  # set new environment with teams
                   encode=(True, True))

        self.opponent_agent = RandomPlayer()

        self.num_resets = -1
        # to do: figure out reward or index
        self.player_index, self.opponent_index = self._get_player_opp_index()
        # if this is needed then do a reset call here. shouldn't be needed
        #self.current_obs_list = [np.zeros((self.env.observation_space.n,))]
        self.current_obs_list = []

        self.reward_range = (-1, 1)
        self.action_space = self.env.action_space
        #self.observation_space = self.env.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                #"state": self.env.observation_space,
                # sheeprl wants a box space
                "state": gym.spaces.Box(low=0, high=1, shape=(self.env.observation_space.n,), dtype=np.float32)
            }
        )

    def step(self, action):
        '''
        Get opponent action then step through env
        '''

        # if opponent needs the non-bool obs can do this
        # may be an unnecessary step
        opponent_obs = self._change_obs_bool_to_float(self.current_obs_list[self.opponent_index])
        opponent_action = self.opponent_agent.get_action(opponent_obs)

        if self.player_index == 0:
            action_list = [action, opponent_action]
        else:
            action_list = [opponent_action, action]

        self.current_obs_list, reward_list, terminated, truncated, info = self.env.step(action_list)

        obs = self.current_obs_list[self.player_index]
        obs = self._change_obs_bool_to_float(obs)
        obs_dict = self._put_obs_in_dict(obs)

        # get custom reward
        reward = self._win_loss_reward(terminated, self.player_index)

        return obs_dict, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        ''''''
        self.num_resets += 1

        # if doing random teams
        #self.env = self._get_random_team_env(self.team_generator)
    
        self.current_obs_list, info = self.env.reset()
    
        self.player_index, self.opponent_index = self._get_player_opp_index()

        obs = self.current_obs_list[self.player_index]
        obs = self._change_obs_bool_to_float(obs)
        obs_dict = self._put_obs_in_dict(obs)

        return obs_dict, info

    def render(self, mode='console'):
        ''''''
        self.env.render(mode)

    def close(self):
        self.env.close()

    def _get_player_opp_index(self):
        '''
        Get the player and opponent index
        return 0 or 1 depending on which team you are
        Used for accessing obs indices, reward indices,
        telling who winner is etc
        '''
        # player_index = self.num_resets % 2
        # opp_index = (player_index + 1) % 2
        # for testing do 0 and 1
        player_index = 0
        opp_index = 1

        return player_index, opp_index
        
    def _win_loss_reward(self, terminated, player_index):
        '''
        Does a reward for winning or losing
        winner is -1 unless a winner has been picked
        '''
        reward = 0.
        if terminated:
            winner = self.env.winner

            if winner == 0 or winner == 1:
                if winner == player_index:
                    reward = 1.
                else:
                    reward = -1.
            #print(f"reward {reward} | terminated {terminated} | winner {self.env.winner} | player_index {player_index}|")
        return reward

    def _change_obs_bool_to_float(self, obs_list):
        '''
        Some values in the obs are bools
        '''
        obs_list = [float(x) if isinstance(x, bool) else x for x in obs_list]
        obs_array = np.array(obs_list, dtype=np.float32).clip(0.0, 1.0)

        return obs_array

    def _put_obs_in_dict(self, obs):
        '''
        Put the obs in a dict
        '''
        obs_dict = {
            "state": obs
        }

        return obs_dict

    def _get_random_team_env(self, team_generator):
        '''
        '''
        team0 = team_generator.get_team().get_battle_team([0, 1, 2])
        team1 = team_generator.get_team().get_battle_team([0, 1, 2])

        env = PkmBattleEnv((team0, team1), encode=(True, True))

        return env
