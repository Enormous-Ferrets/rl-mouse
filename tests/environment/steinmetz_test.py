from mouse.agent.agent import Agent
from mouse.environment.steinmetz import Steinmetz
from mouse.model import Action, Stimulus


class TestSteinmetzEnv:

    #  def test_foo(self):
    #      s = Steinmetz()
    #      s.reset()
    #      print(s._observe())
    #      s.step(Action.LEFT)
    #      print(s._observe())
    #      s.step(Action.LEFT)
    #      print(s._observe())
    #
    #      s.render()

    def test_bar(self):
        #  agent = Agent()
        #  foo = agent._get_screen()
        #  print(foo.shape)

        s = Steinmetz()
        s.reset()
        s._stimulus = Stimulus.right()

        #  s.step(Action.LEFT)
        #  s.step(Action.LEFT)
        #  s.step(Action.LEFT)
        #  s.step(Action.LEFT)
        #  s.step(Action.LEFT)
        s.render()

        print(s.stimulus.is_in_centre())

