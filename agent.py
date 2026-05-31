class Agent:
    def __init__(self, agent_id: int):
        # agent_id: 0, 1, 2, hoặc 3, được tự động gán bởi engine
        self.agent_id = agent_id

    def act(self, obs: dict) -> int:
        pass