from _utils import check_symmetric_decomposable,check_wl_determinable,graph_generator, derive_adjacency,wltest_coloring
import _utils
from gurobipy import Model
import json
from tabulate import tabulate
import os


def generate_domains(outer_path,problem_type,problem_name):
    print('domain path:',f'{outer_path}/{problem_type}/{problem_name}/domains.json')
    with open(f'{outer_path}/{problem_type}/{problem_name}/domains.json', 'r') as f:
        domains = list(json.load(f).keys())
    return domains

def get_problem_lists(outer_path,problem_type,problem_name):
    sym_path_list = []
    sym_decom_list = []
    path_list = []
    wl_det_list =[]
    with open('problem_names.json', 'r', encoding='utf-8') as file:
        selected_problem = json.load(file)
    for domain in list(selected_problem[problem_type][problem_name].keys()):
        for vari in selected_problem[problem_type][problem_name][domain]:
            o_path = f'{problem_type}/{problem_name}'
            path = f'{o_path}/{problem_name}_[{domain}]{vari}/model.lp'   
            print(f'Processing {problem_type}_{problem_name}_[{domain}]{vari}')  
            if os.path.exists(path):
                exist = 1
                path_list.append(f'{problem_name}_[{domain}]{vari}')
            else:
                exist = 0

            if exist == 1:
                info1 = graph_generator(path)
            else:
                continue
        
            A1 = info1[0] 
            f1 = info1[1]#variable
            c1 = info1[2]#constrain
            Adj = derive_adjacency(A1)
            color = wltest_coloring(c1,f1,A1)[0]
            if check_wl_determinable(color)==False:
                #print(f'{problem_name}_[{domain}]{vari}:Symmetric')
                sym_path_list.append(f'{problem_name}_[{domain}]{vari}')
                if check_symmetric_decomposable(color,Adj):
                    sym_decom_list.append(f'{problem_name}_[{domain}]{vari}')
            else: 
                wl_det_list.append(f'{problem_name}_[{domain}]{vari}')
    return path_list,sym_path_list,sym_decom_list,wl_det_list


def generate_valid_problem_list(outer_path,problem_type):
    all_path_list = []
    with open(f'{outer_path}/{problem_type}/{problem_type}_name.json', 'r') as f:
        problem_name_list = list(json.load(f).keys())
 
    for problem_name in problem_name_list:
        path_dict = {}
        path_list,sym_path_list,sym_decom_list,wl_det_list = get_problem_lists(outer_path,problem_type,problem_name)
        path_dict[problem_name] = {'path_list':path_list,'sym_path_list':sym_path_list,'sym_decom_list':sym_decom_list,'wl_det_list':wl_det_list}
        all_path_list.append(path_dict)
        data = [
        [f'Total', len(path_list)],
        [f'Symmetric', len(sym_path_list)],
        [f'Symmetric Decomposable', len(sym_decom_list)],
        [f'WL-Determinable', len(wl_det_list)]]
        print(f"******************************{problem_type}********************************************")
        print(tabulate(data, headers=[f"{problem_name}", "Count"], tablefmt="grid"))
        print("**************************************************************************")

    with open(f"valid_problem_list/{problem_type}_valid_problem_list.json", "w", encoding="utf-8") as f:
        json.dump(all_path_list, f, ensure_ascii=False, indent=4)


outer_path = os.getcwd()
for problem_type in ['LP','MILP']:
    generate_valid_problem_list(outer_path,problem_type)

def main():
    outer_path = os.getcwd()
    problem_types = ['MILP','LP']
    for problem_type in problem_types:
        generate_valid_problem_list(outer_path,problem_type)

if __name__ == '__main__':
    main()
