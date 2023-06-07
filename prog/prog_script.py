import json
import re
import sys
import os
sys.path.append("..")
from utils.exec_utils import grounded_exec_v2
# with open("../prompt/progprompt.json") as f:
#     task_dict = json.load(f)
graph_num, exp_times = sys.argv[1], sys.argv[2]
print(f"experiment time{exp_times} on environment scene{graph_num}")
envname = f"TrimmedTestScene{graph_num}_graph"
with open(f'./progplans/{envname}/exp_{exp_times}.json', "r") as f:
    task_dict = json.load(f)
task_scripts = {}

def prog2script(line):
    res = re.search(r'([a-z]*)\(\'([a-z_]*)\'\)', line)
    resspace = re.search(r'([a-z]*)\(\'([a-z_ ]*)\'\)', line)
    if res is not None:
        action, object = res.group(1).capitalize(), res.group(2)
        return f"[{action}] <{object}> (1)"
    elif resspace is not None:
        action, object = re.search(r'([a-z]*)\(\'([a-z_ ]*)\'\)', line).group(1).capitalize(), re.search(r'([a-z]*)\(\'([a-z_ ]*)\'\)', line).group(2)
        return f"[{action}] <{object.replace(' ', '_')}> (1)"
    else:
        res = re.search(r'([a-z]*)\(\'([a-z_]*)\', \'([a-z_]*)\'\)', line)
        if res is not None:
            action, object1, object2 = res.group(1).capitalize(), res.group(2), res.group(3)
            return f"[{action}] <{object1}> (1) <{object2}> (1)"
        else:
            print("none", line)
            return None


for k, v in task_dict.items():
    assert_script = []
    script = []
    generation = v['plan'][0][0]
    task_name = v['ori_data_item']['task']
    generation_lines = generation.split("\n")
    for line in generation_lines:
        line = line[1:]
        if "#" not in line:
            if line.startswith("assert"):
                assert_script.append(line)
            elif line.startswith("else"):
                line = prog2script(line)
                assert_script.append(f"else: {line}")
            else:
                res = re.search(r'^([a-z]*)\(\'([a-z_]*)\'\)', line)
                if res is not None:
                    action, object = res.group(1).capitalize(), res.group(2)
                    script.append(f"[{action}] <{object}> (1)")
                    assert_script.append(f"[{action}] <{object}> (1)")
                else:
                    res = re.search(r'^([a-z]*)\(\'([a-z_]*)\', \'([a-z_]*)\'\)', line)
                    if res is not None:
                        action, object1, object2 = res.group(1).capitalize(), res.group(2), res.group(3)
                        script.append(f"[{action}] <{object1}> (1) <{object2}> (1)")
                        assert_script.append(f"[{action}] <{object1}> (1) <{object2}> (1)")


    task_scripts[task_name] = {"script": script, "assert_script": assert_script}

if not os.path.exists(f"./progscript/{envname}/"):
    os.makedirs(f"./progscript/{envname}/")

with open(f"./progscript/{envname}/exp_{exp_times}.json", "w") as f:
    f.write(json.dumps(task_scripts, ensure_ascii=False, indent=1))
# with open("../prompt/script2.json", "w") as f:
#     f.write(json.dumps(task_scripts, ensure_ascii=False, indent=1))
    #json.dump(task_scripts, f)

for k, v in task_scripts.items():
    print(k, v)



