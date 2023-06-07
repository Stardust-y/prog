import copy
import json
import os.path
import sys
sys.path.append("..")
from generation.generator import Generator

graph_num, exp_times = sys.argv[1], sys.argv[2]
print(f"experiment time{exp_times} on environment scene{graph_num}")
dataset_root_dir = "../../dataplace/split_v4/"
dataset_name = f"TrimmedTestScene{graph_num}_graph"
dataset_dir = dataset_root_dir + dataset_name
save_dir = f"./progplans/{dataset_name}/exp_"
if not os.path.exists(f"./progplans/{dataset_name}/"):
    os.makedirs(f"./progplans/{dataset_name}/")

prog_prompt_prefix_dir = "../prompt/split_v4/prog_prompt_dynamic.txt"

with open("../key.txt", 'r') as f:
    keys = [line.strip() for line in f.readlines()]

prog_prompt_prefix = "".join(open(prog_prompt_prefix_dir, 'r').readlines())

g_dict = {}

args = {
    'engine': "text-davinci-003",
    'temperature': 0.8,
    'top_p': 0.95,
    'sampling_n': 1,
    "max_generation_tokens": 1024,
    'stop_tokens': ['\n\n'],
}

class D(object):
   def __init__(self):
       self.__dict__.update(args)

args = D()


generator = Generator(args, keys)

dataset = json.load(open(dataset_dir + "/val.json", "r"))

task2graph = json.load(open(dataset_dir + "/task_to_graph.json", "r"))

for i, task in enumerate(dataset):
    taskname = task["task"]
    graph_dict = task2graph[taskname]
    object_list = list(set([node['class_name'] for node in graph_dict['nodes']]))
    prog_prompt_prefix = prog_prompt_prefix.replace("[objectlist]", str(object_list))
    function_name = f'def {"_".join(taskname.lower().split(" "))}():\n'
    prompt = prog_prompt_prefix + function_name
    g_dict[i] = {
        'plan': [],
        'ori_data_item': copy.deepcopy(task)
    }
    res = generator.generate_one_pass(prompts=[(prompt,)])
    plan = res[0].values()
    usage = res[1]
    g_dict[i]['plan'] = list(plan)[0]
    g_dict[i]['usage'] = usage
    print(g_dict[i]['plan'])
    with open(save_dir + str(exp_times) + ".json", "w") as f:
        f.write(json.dumps(g_dict, ensure_ascii=False, indent=1))

