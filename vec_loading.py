import re
import torch

WORD_PATH = "./glove_vecs/glove.twitter.27B.50d.txt"
RE_WSPACE = r'\s+'


def load_word_vecs():
    vecs = {}
    with open(WORD_PATH, "r", encoding="utf-8") as vectors:
        while True:
            line = vectors.readline().strip()
            if not line:
                break
            tag, vec = re.split(RE_WSPACE, line, maxsplit=1)
            vec = re.split(RE_WSPACE, vec)
            vec_t = torch.empty((len(vec)))

            for i in range(len(vec)):
                vec_t[i] = float(vec[i])

            vecs[tag] = vec_t

    return vecs
