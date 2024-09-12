import streamlit as st
import pandas as pd

def simple_traj(row): 
    steps = list(row["intermediate_steps"])[0]
    outcome = ""
    for t in steps:
        if "observation" in t and "final_answer" not in t:
            for split_obs in t["observation"].split("### 1. Search outcome (short version):")[-1].split("\n"):
                obs = split_obs.strip()
                if obs: break
            outcome += f"**Output:**\n{t['llm_output'].strip()}\n\n"
            # outcome += f"Tool Name: {t['tool_call']['tool_name'].strip()}\n\n" 
            # outcome += f"Tool Args: {t['tool_call']['tool_arguments'].strip()}\n\n" 
            outcome += f"**Observation**: {obs}\n\n"

        elif "final_answer" in t:
            outcome += "**Final Answer**: " + str(t["final_answer"]) + "\n"
    return outcome

def create_graph(data_row):
    # "Agent name: " + data_row["agent_name"].iloc[0],
    st.write("**Question:** " + data_row["question"].iloc[0])
    cols_to_print = [
        "prediction", 
        "parsing_error", 
        "iteration_limit_exceeded",
        "agent_error",
        "task",
        "true_answer"
        ]
    data_row[cols_to_print]
    traj = simple_traj(data_row)
    st.write(traj)
    # steps = list(data_row["intermediate_steps"])[0]
    # graph = graphviz.Digraph()
    # graph.edge("hi", "no")
    # st.graphviz_chart(graph) 
    return 

file_path = './gpt_4o_mini_test.jsonl' 
jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
create_graph(jsonObj.head(1))