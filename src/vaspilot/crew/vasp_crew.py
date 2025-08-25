from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.utilities import I18N
import os
import copy
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from ..tools.wait_calc_tool import WaitCalcTool
import yaml
from typing import Dict, Any
from .embedding import LocalAPIEmbedder
from crewai_tools import MCPServerAdapter

class VaspCrew():
	"""chatmaterials crew"""

	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.llm_mapper = config['llm_mapper']
		self.llm_config = {}
		for key, value in config['llm_config'].items():
			self.llm_config[key] = LLM(**self.llm_mapper[value])
		self.tool_dicts = {}
		if self.config['mcp_server'] is not None:
			mcp_params = copy.deepcopy(self.config['mcp_server'])
			self.mcp_server = MCPServerAdapter(mcp_params)
		for tools in self.mcp_server.tools:
			self.tool_dicts[tools.name] = tools
		self.tool_dicts["wait_calc_tool"] = WaitCalcTool(mcp_url=self.config['mcp_server']['url'])
		self.embedder = LocalAPIEmbedder(url=config["embbeder"]["url"], model_id=config["embbeder"]["model_id"], api_key=config["embbeder"]["api_key"])

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
		
		if self.config['agents'][agent_name]['tools']:
			tools = [self.tool_dicts[tool_name] for tool_name in self.config['agents'][agent_name]['tools']]
		else:
			tools = []
		
		fn_call_llm = self.llm_config.get('fn_call_llm', None)
		
		return Agent(
			role=agent_name,
			goal=self.config['agents'][agent_name]['goal'],
			backstory=self.config['agents'][agent_name]['backstory'],
			llm = self.llm_config[agent_name],
			tools = tools,
			function_calling_llm=fn_call_llm,
		)

	
	def crew(self, work_dir: str) -> Crew:
		"""Creates the ChatMaterials crew"""
		if not os.path.exists(f"{work_dir}/memory/"):
			os.makedirs(f"{work_dir}/memory/")
		return Crew(
			agents=[self.create_agent(agent_name) for agent_name in self.config['agents'].keys()],
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
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
