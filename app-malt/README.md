# Capacity planning in datacenter
## Run the application
Example usage.
```python
python main.py --llm_agent_type AzureGPT4Agent --num_queries 3 --complexity_level level1 level2 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl
```

```python
python main.py --llm_agent_type Qwen/Qwen2.5-72B-Instruct --num_queries 3 --complexity_level level1 --output_dir logs/llm_agents --output_file qwen.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl
```

## Code structure
`dy_query_generation.py` generate the dynamic benchmark datset. Input: number of queries per cataorgory, complexity level; Output: benchmark_data.jsonl

`solid_step_helper.py` contains all the functions that help dynamically generating ground truth for new queries.

`error_check.py` check if LLM generated answer satisify all the safety constraints.

`llm_model.py` includes all the LLM agents and their prompt used.

`malt_env.py` is the application simulator. It takes the LLM generated answer, run it in the enviroment and return the evaluated results.

`main.py` is the end-to-end controller. It first generate a new set of benchmark, second run the LLM agent, third analyze the results.

## LLM usage
### Azure GPT
Obtain GPT resources and endpoints
If you use Azure GPT on a Azure VM, need to use the following
```python
from azure.identity import AzureCliCredential
# Get the Azure Credential
credential = AzureCliCredential()
```
Otherwise, use the following
```python
from azure.identity import DefaultAzureCredential
# Get the Azure Credential
credential = DefaultAzureCredential()
```
And please update the below with your own endpoint information
```python
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("please_update").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "please_update"
```



