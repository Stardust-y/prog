# delete illegal actions before execute
import sys
import os.path
sys.path.append("..")
sys.path.append("../..")
from simulation.evolving_graph.scripts import read_script, script_to_list_string, read_script_from_list_string, read_script_from_string
from prog_exec import Script
from utils.exec_utils import exec_one_step, prepare_for_execution
import json
graph_num, exp_times = sys.argv[1], sys.argv[2]
print(f"experiment time{exp_times} on environment scene{graph_num}")
envname = f"TrimmedTestScene{graph_num}_graph"
with open(f"./progscript/{envname}/exp_{exp_times}.json") as f:
    scripts = json.load(f)

with open(f"../../dataplace/split_v4/{envname}/task_to_graph.json") as f:
    graph_dicts = json.load(f)

if not os.path.exists(f"./progpreprocessed/{envname}/"):
    os.makedirs(f"./progpreprocessed/{envname}/")

processed_script = {}

task_names = list(scripts.keys())
script_str_lists = [v['script'] for k, v in scripts.items()]
assert_script_str_lists = [v['assert_script'] for k, v in scripts.items()]

for i, (script_str_list, assert_script_str_list) in enumerate(zip(script_str_lists, assert_script_str_lists)):
    print(task_names[i])
    graph_dict = graph_dicts[task_names[i]]
    script, assert_script = [], []
    auto_script_origin = Script(script_str_list, assert_script_str_list)
    next_step = ""
    print("script before", script_str_list)
    while "END" not in next_step:
        next_step = auto_script_origin.next_step()
        try:
            if "else" in next_step:
                else_split = next_step.split("else: ")
                step_list = read_script_from_list_string([else_split[k+1] for k in range(len(else_split)-1)])
            else:
                step_list = read_script_from_list_string([next_step])
            prepare_for_execution(graph_dict, step_list)
            step_str_list_aligned = script_to_list_string(step_list)
        except Exception as e:
            print("illegal action", e, " be deleted")
        else:
            if "assert" in next_step:
                assert_script.append(next_step.split("else")[0])
                for j in range(len(next_step.split("else")) - 1):
                    assert_script.append("else" + next_step.split("else")[j+1])
            else:
                assert_script.append(next_step)
                script.append(next_step)
    processed_script[str(i)] = {"script": script, "assert_script": assert_script}

    with open(f"./progpreprocessed/{envname}/exp_{exp_times}.json", "w") as f:
        f.write(json.dumps(processed_script, ensure_ascii=False, indent=1))


