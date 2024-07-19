from gymnasium.envs.registration import register

register(
     id="PkmBattleEnv-v0",
     entry_point="vgc.engine.PkmBattleEnv:PkmBattleEnv"
)