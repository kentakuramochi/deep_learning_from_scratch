import sys
sys.path.append("..")

import numpy as np
from common.functions import softmax
from ch6.rnnlm import Rnnlm
from ch6.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    """ Sentence generator by RNN LM
    """
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]  # The first word id

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)  # Output of RNNLM (before normalization)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)

            # Skip specified word id,
            # like <unk> (rare word), N (number) in PTB dataset
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    """ Sentence generator by better RNN LM
    """
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]  # The first word id

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)  # Output of RNNLM (before normalization)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)

            # Skip specified word id,
            # like <unk> (rare word), N (number) in PTB dataset
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
