from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.utilities import I18N
from crewai.tools.agent_tools.agent_tools import AgentTools
import os
import copy
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from ..tools.wait_calc_tool import WaitCalcTool
from ..tools.json_rag_tool import JsonApproxSearch, JsonStrictSearch
from crewai_tools import RagTool
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
import yaml
from typing import Dict, Any
from .embedding import LocalAPIEmbedder
from crewai_tools import MCPServerAdapter

class VaspCrew():
	"""VASPilot crew"""

	def __init__(self, config: Dict[str, Any]):
		self.config = copy.deepcopy(config)
		self.llm_mapper = config['llm_mapper']
		self.llm_config = {}
		for key, value in config['llm_config'].items():
			self.llm_config[key] = LLM(**self.llm_mapper[value])
		self.embedder = LocalAPIEmbedder(url=config["embbeder"]["url"], model_id=config["embbeder"]["model_id"], api_key=config["embbeder"]["api_key"])
		self.tool_dicts = {}
		if self.config['mcp_server'] is not None:
			mcp_params = copy.deepcopy(self.config['mcp_server'])
			self.mcp_server = MCPServerAdapter(mcp_params)
		for tools in self.mcp_server.tools:
			self.tool_dicts[tools.name] = tools
		self.tool_dicts["wait_calc_tool"] = WaitCalcTool(mcp_url=self.config['mcp_server']['url'])
		if self.config.get("tool_params", None).get("json_approx_search_tool", None):
			ragtool = JsonApproxSearch(embedding_function=self.embedder)
			detailtool = JsonStrictSearch()
			if self.config['tool_params']['json_approx_search_tool'].get("sources", None):
				for data_path in self.config['tool_params']['json_approx_search_tool']['sources']:	
					ragtool.add(json_file_path=data_path)
			if self.config['tool_params']['json_strict_search_tool'].get("sources", None):
				for data_path in self.config['tool_params']['json_strict_search_tool']['sources']:
					detailtool.add(json_file_path=data_path)
			self.tool_dicts["json_approx_search_tool"] = ragtool
			self.tool_dicts["json_strict_search_tool"] = detailtool
		self.agent_dict = {}
		self.agent_tools_dict = {"ask_question_tool":[], "delegate_work_tool":[]}
		

	def stop(self):
		self.mcp_server.stop()

	def create_manager_agent(self) -> Agent:
		manager_goal = self.config['agents']['manager_agent']['goal']
		manager_backstory = self.config['agents']['manager_agent']['backstory']
		manager = Agent(
			role="Project Manager",
			goal=manager_goal, #"Efficiently manage the crew and ensure high-quality task completion",
			backstory=manager_backstory, #"You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
			allow_delegation=True,
			llm=self.llm_config["manager"],
			function_calling_llm=self.llm_config["fn_call_llm"],
		)
		return manager

	def create_agent(self, agent_name: str) -> Agent:
		if not self.config['agents'][agent_name]:
			raise ValueError(f"Agent {agent_name} not found in config")
		
		tools = []
		if self.config['agents'][agent_name].get('tools', None):
			for tool_name in self.config['agents'][agent_name]['tools']:
				if tool_name == "ask_question_tool" or tool_name == "delegate_work_tool":
					self.agent_tools_dict[tool_name].append(agent_name)
				else:
					tools.append(self.tool_dicts[tool_name])
		
		fn_call_llm = self.llm_config.get('fn_call_llm', None)

		if self.config['agents'][agent_name].get('role', None):
			role = self.config['agents'][agent_name]['role']
		else:
			role = agent_name

		self.agent_dict[agent_name] = Agent(
			role=role,
			goal=self.config['agents'][agent_name]['goal'],
			backstory=self.config['agents'][agent_name]['backstory'],
			llm = self.llm_config[agent_name],
			tools = tools,
			function_calling_llm=fn_call_llm,
		)

		return self.agent_dict[agent_name]

	def create_working_agents(self, agent_names: list[str]) -> list[Agent]:
		agents = []
		for agent_name in agent_names:
			if agent_name == 'manager_agent':
				continue
			agents.append(self.create_agent(agent_name))
		return agents

	def crew(self, work_dir: str) -> Crew:
		"""Creates the ChatMaterials crew"""
		if not os.path.exists(f"{work_dir}/memory/"):
			os.makedirs(f"{work_dir}/memory/")
		working_agents = self.create_working_agents(self.config['agents'].keys())
		asked_agents = []
		delegated_agents = []
		if self.config.get("tool_params", None).get("ask_question_tool", None):
			for agents in self.config["tool_params"]["ask_question_tool"].get("agents", None):
				asked_agents.append(self.agent_dict[agents])
		if self.config.get("tool_params", None).get("delegate_work_tool", None):
			for agents in self.config["tool_params"]["delegate_work_tool"].get("agents", None):
				delegated_agents.append(self.agent_dict[agents])
		_, ask_question_tool = AgentTools(agents=asked_agents).tools()
		delegate_work_tool,_  = AgentTools(agents=delegated_agents).tools()
		for agent_name in self.agent_tools_dict["ask_question_tool"]:
			self.agent_dict[agent_name].tools.append(ask_question_tool)
		for agent_name in self.agent_tools_dict["delegate_work_tool"]:
			self.agent_dict[agent_name].tools.append(delegate_work_tool)
		return Crew(
			agents=working_agents,
			tasks=[],
			process=Process.hierarchical,
			verbose=True,
			output_log_file=f"{work_dir}/output.log",
			function_calling_llm=self.llm_config["fn_call_llm"],
			#planning=True,
			#planning_llm=self.llm_config["planning"],
			manager_agent=self.create_manager_agent(),
			memory=True,
			long_term_memory = LongTermMemory(
        		storage=LTMSQLiteStorage(
            		db_path=f"{work_dir}/memory/ltm_storage.db",
        		)
    		),
			short_term_memory = ShortTermMemory(
				storage = RAGStorage(
					type="short_term",
					path=f"{work_dir}/memory/stm",
					embedder_config={
						"provider": "custom",
						"config": {
							"embedder": self.embedder
						}
    				}
				),
			),
			entity_memory = EntityMemory(
				storage=RAGStorage(
					type="short_term",
					path=f"{work_dir}/memory/etm",
					embedder_config={
						"provider": "custom",
						"config": {
							"embedder": self.embedder
						}
    				}
				),
    		),
		)
