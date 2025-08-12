from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.utilities import I18N
import os
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.memory.storage.rag_storage import RAGStorage
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from ..tools.wait_calc_tool import WaitCalcTool
import yaml
from typing import Dict, Any
from .embedding import LocalAPIEmbedder
from crewai_tools import MCPServerAdapter

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
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
			self.mcp_server = MCPServerAdapter(self.config['mcp_server'])
		for tools in self.mcp_server.tools:
			self.tool_dicts[tools.name] = tools
		self.tool_dicts["wait_calc_tool"] = WaitCalcTool(mcp_url=self.config['mcp_server']['url'])
		self.embedder = LocalAPIEmbedder(url=config["embbeder"]["url"], model_id=config["embbeder"]["model_id"], api_key=config["embbeder"]["api_key"])

	def stop(self):
		self.mcp_server.stop()

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def crystal_structure_agent(self) -> Agent:
		return Agent(
			role="Crystal Structure Agent",
			goal=self.config['agents']['crystal_structure_agent']['goal'],
			backstory=self.config['agents']['crystal_structure_agent']['backstory'],
			llm = self.llm_config["crystal_structure_agent"],
			tools = [self.tool_dicts[tool_name] for tool_name in self.config['agents']['crystal_structure_agent']['tools']],
			function_calling_llm=self.llm_config['fn_call_llm'],
		)
	@agent
	def vasp_agent(self) -> Agent:
		return Agent(
			role="VASP Agent",
			goal=self.config['agents']['vasp_agent']['goal'],
			backstory=self.config['agents']['vasp_agent']['backstory'],
			llm = self.llm_config["vasp_agent"],
			tools = [self.tool_dicts[tool_name] for tool_name in self.config['agents']['vasp_agent']['tools']],
			function_calling_llm=self.llm_config['fn_call_llm']
		)

	@agent
	def result_validation_agent(self) -> Agent:
		return Agent(
			role="Result Validation Agent",
			goal=self.config['agents']['result_validation_agent']['goal'],
			backstory=self.config['agents']['result_validation_agent']['backstory'],
			llm = self.llm_config["result_validation_agent"],
			tools = [self.tool_dicts[tool_name] for tool_name in self.config['agents']['result_validation_agent']['tools']],
			function_calling_llm=self.llm_config['fn_call_llm']
		)

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

	@crew
	def crew(self, work_dir: str) -> Crew:
		"""Creates the ChatMaterials crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge
		if not os.path.exists(f"{work_dir}/memory/"):
			os.makedirs(f"{work_dir}/memory/")
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
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
