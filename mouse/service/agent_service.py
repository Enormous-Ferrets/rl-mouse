from pathlib import Path

import torch

from mouse.agent.agent import Agent


class AgentService:

    @staticmethod
    def train():
        pass

    @staticmethod
    def infer(agent: Agent):
        pass

    @staticmethod
    def load(path: Path) -> Agent:
        agent = Agent.from_weights(path)

        return agent
