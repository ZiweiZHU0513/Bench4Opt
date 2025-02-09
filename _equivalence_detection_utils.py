from _utils import stable_partition
import json


def check_color_equivalence(color1,color2):
    samecolor_cluster1 = stable_partition(color1)
    samecolor_cluster2 = stable_partition(color2)
    for i in samecolor_cluster1.keys():
        if i not in samecolor_cluster2.keys():
            print('-------------An unique color occurs in color1 but not in color2-----------')
            return False
        else:
            if len(samecolor_cluster1[i]) != len(samecolor_cluster2[i]):
                print('-----------An unmatched cluster occurs!---------')
                return False
    return True


def check_answer_name(problem_name,path,problem_type):
    with open(f"valid_problem_list/{problem_type}_valid_problem_list.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for problem in data:
        if problem_name in problem.keys():
            if path in problem[problem_name]['path_list']:
                if path in problem[problem_name]['sym_decom_list']:
                    return 'Symmetric Decomposable'
                else:
                    return 'WL-determinable'
            else:
                return 'path not found'

def get_valid_path_list(path_list,sym_path_list,sym_decom_list):
    valid_path_list = []
    for path in path_list:
        if path not in sym_path_list:
            valid_path_list.append(path)
        elif path in sym_decom_list:
            valid_path_list.append(path)
    return valid_path_list