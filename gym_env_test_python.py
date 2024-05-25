'''
python gym_env_test_python.py
'''

from vgc.datatypes.Objects import PkmTeam, Pkm, GameState, Weather
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.behaviour.BattlePolicies import RandomPlayer, TerminalPlayer

def main():

    team0, team1 = PkmTeam(), PkmTeam()
    agent0, agent1 = RandomPlayer(), RandomPlayer()
    env = PkmBattleEnv((team0, team1),
                    encode=(agent0.requires_encode(), agent1.requires_encode()))  # set new environment with teams
    n_battles = 1  # total number of battles
    t = False
    battle = 0
    while battle < n_battles:
        s, _ = env.reset()
        while not t:  # True when all pkms of one of the two PkmTeam faint
            a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
            s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
            env.render()
        t = False
        battle += 1
    print(env.winner)  # winner id number


if __name__ == "__main__":
    main()