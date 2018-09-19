from itertools import chain
import numpy as np
def params_append(list_params_left, list_param_right):
    if type(list_params_left) is not list:
        list_params_left = list(map(lambda p: list([p]), list_params_left))
    n_left = len(list_params_left)
    n_right = len(list_param_right)
    list_params_left *= n_right
    list_params_left = [x[:] for x in list_params_left]
    list_param_right = list(chain([[p] * n_left for p in list_param_right]))
    list_param_right = np.array(list_param_right).flatten()
    print(list_params_left, list_param_right, sep= '\n')
    list_params_left[4].append(2)
    print(list_params_left)
    for i in range(len(list_params_left)):
        list_params_left[i].append(list_param_right[i])
    return list_params_left


def get_grid_params(search_params):
    """遍历 grid search 的所有参数组合。
    Args:
        search_params: dict of params to be search.
         search_params = {'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.20],
                             'max_depth': [4, 5, 6, 7],
                             'colsample_bytree': [0.6, 0.7, 0.8]}
    Returns:
        grid_params: list, 每个元素为一个dict, 对应每次搜索的参数。
    """
    keys = list(search_params.keys())
    values = list(search_params.values())
    grid_params = list()
    if len(keys) == 1:
        for value in values[0]:
            dict_param = dict()
            dict_param[keys[0]] = value
            grid_params.append(dict_param.copy())
        return grid_params
    list_params_left = [[p] for p in values[0]]
    for i in range(1, len(values)):
        list_param_right = values[i]
        list_params_left = params_append(list_params_left, list_param_right)
    for params in list_params_left:
        dict_param = dict()
        for i in range(len(keys)):
            dict_param[keys[i]] = params[i]
        grid_params.append(dict_param.copy())
    return grid_params

if __name__  == '__main__':
    search_params = {'num_leaves':[20, 30, 40, 50],
                     'learning_rate': [0.025, 0.05, 0.1, 0.15, 0.20]}
    m = get_grid_params(search_params)
    print(len(m))