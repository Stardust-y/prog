import copy
import json
import os.path
import sys
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from utils.exec_utils import exec_one_step, prepare_for_execution
from evolving_graph.environment import EnvironmentGraph, EnvironmentState
from evolving_graph.scripts import read_script, script_to_list_string, read_script_from_list_string, read_script_from_string
from evolving_graph.execution import ScriptExecutor
import evolving_graph.utils as utils
from arguments import get_args
from utils.deciding_graph import Deciding_Graph
from utils.env_utils import _mask_state, observation_prompt, get_env_prompt, prog_observation_prompt, prog_rule_prompt
from transformers import AutoTokenizer
import openai
from utils.retriever import Retriever
from utils.exec_utils import calc_gcr
from grounded_deciding import show_result
from utils.retriever import Retriever
from sampling_grounding_deciding.utils.generator import generate


class Script:
    def __init__(self, script_str_list, assert_script_str_list):
        self.script = script_str_list
        self.assert_script = assert_script_str_list
        self.p_script = 0
        self.p_assert = 0

    def next_step(self):
        if self.p_assert - len(self.assert_script) >= 0:
            return "[END]"
        elif "assert" in self.assert_script[self.p_assert]:
            exec_assert = self.assert_script[self.p_assert] + self.assert_script[self.p_assert + 1]
            self.p_assert += 2
            while self.p_assert - len(self.assert_script) < 0 and "else" in self.assert_script[self.p_assert]:
                exec_assert += self.assert_script[self.p_assert]
                self.p_assert += 1
            return exec_assert
        else:
            exec_action = self.script[self.p_script]
            self.p_assert += 1
            self.p_script += 1
            return exec_action

def prog_exec(args, script_str_list, assert_script_str_list, graph_dict, retriever, verbose=False):
    max_retry_times, retry_cnt = 0, 0  # when retry_times == 0, do not do error correction
    graph = EnvironmentGraph(graph_dict)
    name_equivalence = utils.load_name_equivalence()
    executor = ScriptExecutor(graph, name_equivalence)
    state = EnvironmentState(executor.graph, executor.name_equivalence, instance_selection=True)
    graph_state_list, state_traceback = [], []
    token_num = 0
    try:
        script_list = read_script_from_list_string(script_str_list)
        prepare_for_execution(graph_dict, script_list)
        script_str_list_aligned = script_to_list_string(script_list)
        print(script_str_list_aligned)
        auto_script = Script(script_str_list_aligned, assert_script_str_list)
    except Exception as e:
        print("illegal action", e)
        return False, state, graph_state_list, "have illegal action", token_num
    while True:
        graph_state_list.append(state.to_dict())
        next_step = auto_script.next_step()
        #print("next step", next_step)
        if "[END]" in next_step:
            print("end of script",)
            break
        elif "assert" in next_step:
            print("next step-------", next_step)
            #prompt = prog_observation_prompt(state, next_step.split("else")[0], retriever)
            prompt = prog_rule_prompt(state, next_step.split("else")[0])
            prompt = f"{prompt}, give one word answer True or False to this assert statement: {next_step.split('else')[0]}, "
            response, total_tokens = generate(prompt)
            prediction = True if "True" in response else False
            token_num += total_tokens
            print("prompt", prompt)
            print("prediction", prediction)
            if prediction:
                continue
            else:
                # execute else step
                future_script = read_script_from_string(next_step.split(": ")[1])
                prepare_for_execution(graph_dict, future_script)
                executability, state = exec_one_step(future_script, state, executor)
                if not executability:
                    if verbose: print(executor.info.get_error_string())
                    if retry_cnt >= max_retry_times:
                        print("execute failed at assert else at", script_to_list_string(future_script))
                        end_of_execution = "failed"
                        return False, state, graph_state_list,executor.info.get_error_string(), token_num

        else:
            future_script = read_script_from_string(next_step)
            executability, state = exec_one_step(future_script, state, executor)
            if not executability:
                print("execute failed at", next_step)
                if verbose: print(executor.info.get_error_string())
                if retry_cnt >= max_retry_times:
                    end_of_execution = "failed"
                    return False, state, graph_state_list, executor.info.get_error_string(), token_num
    return True, state, graph_state_list, "", token_num

    # graph_state_list, state_traceback = [], []
    # while True:
    #     graph_state_list.append(state.to_dict())
    #     state_traceback.append(copy.deepcopy(state))


if __name__ == "__main__":
    args = get_args()
    graph_num, exp_times = args.graph_num, args.exp_times
    print(f"experiment time{exp_times} on environment scene{graph_num}")
    envname = f'TrimmedTestScene{graph_num}_graph'
    with open(f"./progpreprocessed/{envname}/exp_{exp_times}.json") as f:
        scripts = json.load(f)

    with open(f"../../dataplace/split_v4/{envname}/task_to_graph.json", "r") as f:
        graph_dicts = json.load(f)
    with open(f"../../dataplace/split_v4/{envname}/val.json", "r") as f:
        val_datasets = json.load(f)
    tasknames = [task["task"] for task in val_datasets]

    with open(f"./progplans/{envname}/exp_{exp_times}.json") as f:
        generation_result = json.load(f)
    if not os.path.exists(f"./progmetrics/{envname}/"):
        os.makedirs(f"./progmetrics/{envname}/")

    script_str_lists = [v['script'] for k, v in scripts.items()]
    assert_script_str_lists = [v['assert_script'] for k, v in scripts.items()]

    retriever = Retriever()

    metrics_dict = {}
    token_num_list = []
    # traverse every script
    for i, (script_str_list, assert_script_str_list) in enumerate(zip(script_str_lists, assert_script_str_lists)):
        graph_dict = graph_dicts[tasknames[i]]
        init_graph_dict = copy.deepcopy(graph_dict)
        plan_token_nums = generation_result[str(i)]["usage"]["total_tokens"]
        if len(script_str_list) == 0:
            print(f"script {i} has illegal [action]()")
            executability, sr, gcr = False, 0, 0.0
            metrics = {"executability": int(executability), "success rate": sr, "gcr": gcr, "tokens": plan_token_nums}
            metrics_dict[i] = metrics
            continue

        executability, state, graph_state_list, info, token_num = prog_exec(
            args=args,
            script_str_list=script_str_list,
            assert_script_str_list=assert_script_str_list,
            graph_dict=graph_dict,
            retriever=retriever
        )
        if executability:
            gcr = calc_gcr(init_graph_dict, state.to_dict(), generation_result[str(i)]['ori_data_item']['goal_condition'])
            sr = 1 if abs(gcr - 1) < 1e-10 else 0
            print("gcr, sr", gcr, sr)
        else:
            gcr, sr = 0.0, 0
        metrics = {"executability": int(executability), "success rate": sr, "gcr": gcr, "tokens": token_num + plan_token_nums}
        metrics_dict[i] = metrics
        with open(f"./progmetrics/{envname}/exp_{exp_times}.json", "w") as f:
            f.write(json.dumps(metrics_dict, ensure_ascii=False, indent=1))



