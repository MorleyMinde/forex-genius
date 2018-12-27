from btgym import BTgymBaseStrategy
import math
import numpy as np

class MyStrategy(BTgymBaseStrategy):
    """
    Example subclass of BT server inner computation strategy.
    """
    def get_reward(self):
        """
        Shapes reward function as normalized single trade realized profit/loss,
        augmented with potential-based reward shaping functions in form of:
        F(s, a, s`) = gamma * FI(s`) - FI(s);

        - potential FI_1 is current normalized unrealized profit/loss;
        EXCLUDED/ - potential FI_2 is current normalized broker value.
        EXCLUDED/ - FI_3: penalizing exposure toward the end of episode

        Paper:
            "Policy invariance under reward transformations:
             Theory and application to reward shaping" by A. Ng et al., 1999;
             http://www.robotics.stanford.edu/~ang/papers/shaping-icml99.pdf
        """

        # All sliding statistics for this step are already updated by get_state().

        # Potential-based shaping function 1:
        # based on potential of averaged profit/loss for current opened trade (unrealized p/l):

        unrealised_pnl = np.asarray(self.broker_stat['unrealized_pnl'])
        current_pos_duration = self.broker_stat['pos_duration'][-1]

        log = ""
        # We want to estimate potential `fi = gamma*fi_prime - fi` of current opened position,
        # thus need to consider different cases given skip_fame parameter:
        if current_pos_duration == 0:
            # Set potential term to zero if there is no opened positions:
            f1 = 0

        else:
            if current_pos_duration < self.p.skip_frame:
                fi_1 = 0
                fi_1_prime = np.average(unrealised_pnl[-current_pos_duration:])

            elif current_pos_duration < 2 * self.p.skip_frame:
                fi_1 = np.average(
                    unrealised_pnl[-(self.p.skip_frame + current_pos_duration):-self.p.skip_frame]
                )
                fi_1_prime = np.average(unrealised_pnl[-self.p.skip_frame:])

            else:
                fi_1 = np.average(
                    unrealised_pnl[-2 * self.p.skip_frame:-self.p.skip_frame]
                )
                fi_1_prime = np.average(unrealised_pnl[-self.p.skip_frame:])
                log = "unrealised_pnl: {} skip_frame: {} unrealised_pnl[-self.p.skip_frame:]: {} fi_1_prime: {}".format(unrealised_pnl,self.p.skip_frame,unrealised_pnl[-self.p.skip_frame:],fi_1_prime)

            # Potential term:
            f1 = self.p.gamma * fi_1_prime - fi_1

        # Main reward function: normalized realized profit/loss:
        realized_pnl = np.asarray(self.broker_stat['realized_pnl'])[-self.p.skip_frame:].sum()

        # Weights are subject to tune:
        self.reward = (10.0 * f1 + 10.0 * realized_pnl) * self.p.reward_scale

        self.reward = np.clip(self.reward, -self.p.reward_scale, self.p.reward_scale)
        if(math.isnan(self.reward)):
            print(log)
            print(f1)
            return 0.0
        return self.reward
    def get_my_state(self):
        my_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.time_dim)),
                np.frombuffer(self.data.high.get(size=self.time_dim)),
                np.frombuffer(self.data.low.get(size=self.time_dim)),
                np.frombuffer(self.data.close.get(size=self.time_dim)),
                np.frombuffer(self.data.volume.get(size=self.time_dim)),
            )
        ).T

        return my_state