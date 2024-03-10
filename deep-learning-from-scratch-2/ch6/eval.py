import sys
sys.path.append("..")

from rnnlm import Rnnlm
from better_rnnlm import BetterRnnlm
from dataset import ptb
from common.util import eval_perplexity


def eval_rnnlm(model):
    model.load_params()

    corpus, _, _ = ptb.load_data("test")

    model.reset_state()

    ppl_test = eval_perplexity(model, corpus)
    print("Test perplexity: ", ppl_test)


def main():
    print("----- RNN LM -----")
    eval_rnnlm(Rnnlm())

    print("----- Better RNN LM -----")
    eval_rnnlm(BetterRnnlm(wordvec_size=650, hidden_size=650))


if __name__ == "__main__":
    main()
