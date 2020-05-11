# coding=utf-8
from collections import defaultdict


def demo1():
    """
    :return:
    """
    score = {'a': 10, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0}
    neighborship = {'a': {'b': 1.0},
                    'b': {'a': 0.5, 'c': 0.5},
                    'c': {'b': 0.5, 'd': 0.5},
                    'd': {'c': 0.5, 'e': 0.5},
                    'e': {'d': 0.5, 'f': 0.5},
                    'f': {'e': 1.0}}

    # neighborship = {'a': {'b': 0.5, 'a': 0.5},
    #                 'b': {'a': 0.5, 'c': 0.5},
    #                 'c': {'b': 0.5, 'd': 0.5},
    #                 'd': {'c': 0.5, 'e': 0.5},
    #                 'e': {'d': 0.5, 'f': 0.5},
    #                 'f': {'e': 0.5, 'f': 0.5}}

    max_iter = 100
    # 阻尼为 0 时, 结果来回震荡, 不能收敛. (就好像盆子里的水, 没有能量损耗, 不能平静).
    damping = 0.2

    prev_score = score
    this_score = defaultdict(float)
    for i in range(max_iter):
        for node in score.keys():
            this_score[node] += prev_score[node] * damping
            for neighbor, weght in neighborship[node].items():
                this_score[neighbor] += prev_score[node] * (1 - damping) * weght
        prev_score = this_score
        this_score = defaultdict(float)
    print(prev_score)
    return


if __name__ == '__main__':
    demo1()
