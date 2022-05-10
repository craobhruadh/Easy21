import numpy as np
# from dataclasses import dataclass, field
from collections import namedtuple

Card = namedtuple('Card', ['value', 'color'])


# Note: I tried to get fancy but the stated problem as is involves
# one dealer and one player.  If this changes (e.g. multiple players)
# then dataclass is great

# @dataclass
# class Action:
#     actions: list[str] = field(init=False, default_factory=list)

#     def __post_init__(self):
#         self.actions = ["hit", "stick"]

# @dataclass
# class State:
#     dealers_sum: int
#     players_sum: int
#     terminal: bool = False


class Easy21:

    def __init__(self, discount_factor=1,):
        self.discount_factor = 1
        self.setup_new_game()

    def setup_new_game(self):
        """Sets up a new game.  Called at init, can also be
        called to reset Easy21 to initial state"""
        card = self.draw_card(black_only=True)
        self.dealers_sum = card.value
        card = self.draw_card(black_only=True)
        self.players_sum = card.value
        self.terminal = False

    def draw_card(self, black_only=False):
        if black_only:
            color = 'black'
        else:
            i = np.random.randint(0, 3)
            color = 'red' if i == 0 else 'black'
        card = Card(
            value=np.random.randint(1, 11),
            color=color
        )
        return card

    def get_state(self, verbose=False):
        if verbose:
            print(f'Players Sum: {self.players_sum}, Dealers sum: {self.dealers_sum}')
        return self.players_sum, self.dealers_sum

    def step(self, action='hit', verbose=False):
        """Do a step.  Acceptable states are either 'hit'
        or 'stick'.  We do not use 'fold' like a civilized
        person because we are stuck with the British version

        Question: can we store the state in the object
        like I have implemented here, or is it better to
        pass a state into this function? """
        if self.terminal:
            print('Game has stopped.  Reset board with .setup_new_game() method')
            return

        if action == 'hit':
            card = self.draw_card()
            if card.color == 'red':
                self.players_sum -= card.value
            else:
                self.players_sum += card.value

            if verbose:
                print(f'Player drew {card.color} {card.value}. ')
                print(f'   Current score: {self.players_sum} versus {self.dealers_sum}')

            if ((self.players_sum < 1) or (self.players_sum > 21)):
                self.terminal = True

                if verbose:
                    print(f'Player goes bust with a score of {self.players_sum}')
                return -1
        elif action == 'stick':
            # Dealer's turn
            while self.dealers_sum < 17:
                card = self.draw_card()
                if card.color == 'red':
                    self.dealers_sum -= card.value
                else:
                    self.dealers_sum += card.value
                if verbose:
                    print(f'Dealer drew {card.color} {card.value}. ')
                    print(f'   Current score: {self.players_sum} versus {self.dealers_sum}')

                if ((self.dealers_sum < 1) or (self.dealers_sum > 21)):
                    if verbose:
                        print(f'Dealer goes bust with a score of {self.dealers_sum}')

                    self.terminal = True
                    return 1  # if the dealer busts, the player wins

            self.terminal = True
            if self.players_sum < self.dealers_sum:
                return -1
            elif self.players_sum > self.dealers_sum:
                return 1
            else:
                return 0
        else:
            print("State name not recognized.  Either 'hit' to draw a card or 'stick' to stop drawing")
        return 0
