# Retro-TAMER
Retroactive Feedback Assignment within the TAMER Framework

@TODO:
  - Choosing a domain for Tetris
    - Interested in: state representation
  - Candidate: OpenAI Gym Tetris
  
  - Can we build TAMER off of the state-action representation and environment? Can we map visuals to a particular state in the state-action space?


For Checkpoint 1:
Domain Selection:
  - Tetromino (a Tetris clone) by Al Sweigart al@inventwithpython.com: No existing RL agent / framework to support RL agents
  - Borrow / modify implementation of state-action modeling from RL agents in OpenAI Gym Tetris and TetrisRL
  - Key Considerations:
      - TAMER: 
          - Timing: Need to limit how many actions the RL agent is able to make within a certain amount of time (don't want thousands of decisions per second that the user cannot keep up with; also in line with there being a finite resoluion of decisions in normal Tetris)
              - Implement by buffering game loop execution by a certain amount of time. 
          - Timing: Allow human feedback to come in once per loop (once per action agent has taken)
              - Implement by adding event handling "in series" within the game loop, directly after the agent has taken an action and the screen has refreshed
              - Mapping user feedback to agent actions taken:
                  - Difficult to do: @TODO

Next Steps:
  - Implement TAMER in own module with SGDRegressor for a supervised learner, partial_fit(). Follow paper implementation as close as possible
    - Feature representation: concatenation of previous state and next state to be input into the supervised learner; greedily choose action that maximizes output of SGDRegressor given input of feature vectors.
    - DQN - borrow and adapt implementation from TetrisRL to Tetromino
