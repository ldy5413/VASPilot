# VASPilot

[中文版本](README_zh.md)

**VASP automation and analysis using CrewAI framework with MCP support**

VASPilot is an open-source platform that fully automates VASP workflows via a multi-agent architecture built on the CrewAI framework and a standardized Model Context Protocol (MCP). It handles every stage of a VASP study-from retrieving crystal structures and generating input files to submitting Slurm jobs, parsing error messages, and dynamically adjusting parameters for seamless restarts. 

## Architecture

![Architecture Diagram](figs/Fig_1.png)

VASPilot follows a modular, multi-agent architecture consisting of several key components:

### Core Components

- **CrewAI Framework**: Multi-agent orchestration system that manages specialized AI agents
- **MCP Server**: Model Control Protocol server providing tools for VASP calculations and materials analysis
- **Quart and Flask Web Server**: Web interface for task submission and monitoring
- **Specialized Agents**:
  - **Crystal Structure Agent**: Handles structure search, analysis, and manipulation
  - **VASP Agent**: Manages VASP calculations (relaxation, SCF, NSCF)
  - **Result Validation Agent**: Validates and analyzes calculation results
  - **Manager Agent**: Coordinates tasks between agents using hierarchical process

### Key Features

- **Intelligent Task Management**: AI agents automatically plan and execute complex workflows
- **Materials Project Integration**: Direct access to crystal structures from Materials Project database
- **VASP Automation**: Automated setup and execution of VASP calculations
- **Memory Systems**: Long-term, short-term, and entity memory for context retention
- **Real-time Monitoring**: Web interface for task tracking and result visualization
- **Database Storage**: SQLite-based storage for calculation records and metadata

## Installation

### Prerequisites

- Python ≥ 3.10
- Slurm

### Quick Installation

1. Clone the repository:
```bash
git clone https://github.com/JiaxuanLiu-Arsko/VASPilot.git
cd VASPilot
```

2. Install the package:
```bash
pip install .
```

This will automatically install all required dependencies including CrewAI, FastMCP, PyMatGen, ASE, and other materials science libraries.

## Docker Usage

### Prerequisites for Docker

- Docker Engine (v20.10+)
- Docker Compose (v2.0+)
- POTCAR files for VASP calculations
- Materials Project API key
- LLM and embedding API access

### Quick Start with Docker

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to add your API keys and configurations:
```bash
nano .env
```

3. Create the required directory structure:
```bash
mkdir -p configs data/uploads data/memory data/work data/mcp_work data/record data/downloads
```

4. Copy your configuration files to the `configs/` directory:
```bash
cp examples/1.Basic/configs/* configs/
```

5. Modify the configuration files in `configs/` to match your Docker setup:
   - Update `mcp_config.yaml` to use paths inside the container
   - Update `crew_config.yaml` to use the correct MCP server URL

6. Start the services:
```bash
docker-compose up -d
```

### Docker Configuration

The VASPilot Docker setup includes:

- **MCP Server**: Provides tools for VASP calculations (port 8933)
- **Quart Server**: Async web interface with parallel task support (port 51293)

### Configuration Files for Docker

For Docker usage, you need to adjust the configuration files:

#### configs/mcp_config.yaml
```yaml
#Directories for neccessary files in submitting a VASP jobs. e.g. slurm.sh and vdw_kernel.bindat.
attachment_path: /app/data/attachments
#Directory for storing the work directory.
work_dir: /app/data/mcp_work
#Path of database for storing the mcp tool execution records.
db_path: /app/data/record/record.db
#API key for the Materials Project.
mp_api_key: ${MP_API_KEY}
#Directory for storing the downloaded structures.
structure_path: /app/data/downloads
#Default INCAR settings for different VASP jobs.
VASP_default_INCAR:
  relaxation:
    NCORE: 4
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-6
    AMIN: 0.01
    LCHARG: True
    IBRION: 1
    EDIFFG: -0.005
    NSW: 100
    MAGMOM: "5000*0"
    ISIF: 3
    NELM: 120
    LWAVE: False
    LCHARG: True
  scf_nsoc: 
    NCORE: 4
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-7
    AMIN: 0.01
    LCHARG: True
    LWAVE: True
    NELM: 120
    MAGMOM: "5000*0"
  nscf_nsoc:
    NCORE: 4
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-7
    AMIN: 0.01
    LCHARG: True
    LWAVE: True
    NELM: 120
    MAGMOM: "5000*0"
  scf_soc: 
    NCORE: 4
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-7
    AMIN: 0.01
    LCHARG: True
    LWAVE: True
    LSORBIT: True
    LORBIT: 11
    NELM: 120
    MAGMOM: "5000*0"
  nscf_soc:
    NCORE: 4
    PREC: 'Accurate'
    ISMEAR: 0
    SIGMA: 0.03
    EDIFF: 1e-7
    AMIN: 0.01
    LCHARG: True
    LWAVE: True
    ICHARG: 11
    ISART: 1
    LSORBIT: True
    LORBIT: 11
    NELM: 120
    MAGMOM: "5000*0"
```

#### configs/crew_config.yaml
```yaml
llm_mapper:
  your-model-name:
    base_url: ${LLM_BASE_URL}
    api_key: ${LLM_API_KEY}  
    model: openai/${LLM_MODEL_NAME}
    temperature: 0
  
llm_config:
  #Fill in modelname defined in llm_mapper
  crystal_structure_agent: your-model-name
  vasp_agent: your-model-name
  result_validation_agent: your-model-name
  manager: your-model-name
  fn_call_llm: your-model-name

# Setup embedding model for RAG in memory.
embbeder:
  url: ${EMBEDDER_BASE_URL}
  model_id: ${EMBEDDER_MODEL_ID}
  api_key: ${EMBEDDER_API_KEY}

mcp_server:
  # When using docker-compose, use the service name
  url: ${MCP_SERVER_URL}
  transport: streamable-http

agents:
  crystal_structure_agent:
    # prompt for the agent
    goal: "你的目标是根据需要，利用工具搜索、分析或操作晶体结构。"
    backstory: >
        你是一个经验丰富的晶体结构专家，你擅长利用工具搜索、分析或操作晶体结构。
            1. 调用工具查找结构时，要尽可能完整地指定条件。
            2. 调用工具分析结构时，要注意任务相关的性质。
            3. 调用工具操作结构前，要先仔细计划。
        任务完成后，要以markdown格式给一个简短的报告，给出晶体结构、与任务有关的分析结果（如果有）以及任务执行情况。
        晶体结构要以结构文件的**完整路径**给出，不要以文本形式给出。**不要虚构任何数据，给出的所有数据都要从工具结果中获取！**
    # tools for the agent
    tools: 
      - search_materials_project
      - analyze_crystal_structure
      - create_crystal_structure
      - make_supercell
      - symmetrize_structure

  vasp_agent:
    # prompt for the agent
    goal: "你的目标是理解目前的需要，并根据需要，利用工具执行VASP计算。"
    backstory: >
      你是一个经验丰富的VASP专家，你擅长利用工具提交并监管VASP计算。
        1. 调用工具提交VASP计算任务。
        2. 所有计算任务完成后，根据需要，调用工具画图。
        3. 以markdown格式简短总结计算结果。不需要给出计算路径等信息，要清晰地给出每个计算的**calculation id**，并总结与目标相关的计算结果与分析。
        4. 若画了图，给出图片的**完整路径**。当需要画多个图时，考虑多次调用python_plot工具。
      **不要虚构任何数据，给出的所有数据都要从工具结果中获取！**
    # tools for the agent
    tools: 
      - wait_calc_tool
      - vasp_relaxation
      - vasp_scf
      - vasp_nscf_kpath
      - vasp_nscf_uniform
      - python_plot

  result_validation_agent:
    # prompt for the agent
    goal: 检查计算结果，指出任何未完成的任务和虚假内容。
    backstory: >        
        作为结果检查者，你的核心职责是：
            1. 利用工具检查结果，指出任何未完成的任务和虚假内容。
            2. 不需要提出任何其他建议，只需指出现有报告中未完成的任务和虚假内容。
            3. 当检查无误时，明确指出任务已经成功完成。当报告中有虚假部分或没完成的部分时，明确指出有问题的部分。
            4. 当没有得到具体的文件路径和计算id时，应在报告中特别指出，而不是认为计算没有完成。
    tools:
      - check_files_exist
      - read_calc_results_from_db

  manager_agent:
    # prompt for the agent
    goal: 理解用户需求，并将任务分配给合适的agent。
    backstory: >        
        作为团队领导者，你的核心职责是：
            1. 理解用户的需求，并根据需求，将任务分配给合适的agent。分配任务时，只需指定总体任务，具体的执行步骤由执行者自己决定。
            2. 团队成员分工：
                - Crystal Structure Agent：负责晶体结构的构建、查找、分析和操作。当用户没有指定具体晶体结构的路径时，应先让Crystal Structure Agent查找或创建对应的结构。
                - VASP Agent：执行VASP计算任务,并对相应的计算结果进行画图。**用户要求的所有设置都要如实提供给VASP Agent，不要遗漏。**
                - Result Validation Agent：检查任务是否完成、报告中是否有虚假的部分。
            3. **不要虚构任何数据，给出的所有数据都要从同事结果中获取！**
            4. 输出结果前，一定要让Result Validation Agent检查任务是否完成。你需要将尽可能完整的上下文提供给他，并给出每一个需要检查的文件的路径以及计算的id。
```

### Usage

Once both servers are running via Docker, access the web interface at:
```
http://localhost:51293
```

From the web interface, you can:
- Submit new calculation tasks
- Monitor running calculations
- View calculation history
- Download results and analysis reports

## Usage

VASPilot requires configuration of two main components: the MCP server and the CrewAI server. Follow the basic setup example in `examples/1.Basic/` for initial configuration.

### Prerequisites Configuration

Before starting VASPilot, you need to configure several system-specific parameters:


#### 1.Prepare Directories & Files

Ensure the following directories exists and has been properly configured as in [Example](examples/1.Basic/):
  - `mcp/` : Directories for MCP server
    - `attachment/`: VASP job templates and auxiliary files
        - `slurm.sh`: slurm script for submitting VASP calculations
        - `vdw_kernel.bindat`: vdW kernel files. See https://www.vasp.at/wiki/index.php/Nonlocal_vdW-DF_functionals#Kernel_file_vdw_kernel.bindat for details
    - `work/`: MCP tools working directories
    - `record/`: Directory to store execution record of MCP tools
    - `downloads/`: Directory to store downloaded structure files
    - `uploads/`: User uploaded files
  - `crew_server/` : Directories for web server and crewAI.
    - `work/`: Web server working directories
  - `configs/`: 
    - `crew_config.yaml`: Configuration file for crewAI components.
    - `crew_config_en.yaml`: Configuration file with english prompts.
    - `mcp_config.yaml`: Configuration file for MCP server

#### 2. MCP Server Configuration

Edit `configs/mcp_config.yaml` and configure the following paths according to your system:

```yaml
# Directory for VASP job files (slurm.sh, vdw_kernel.bindat, etc.)
attachment_path: your-path-to-example/mcp/attachment

# Working directory for calculations
work_dir: your-path-to-example/mcp/work

# Database path for calculation records
db_path: your-path-to-example/record/record.db

# Materials Project API key
mp_api_key: your-mp-api-key

# Directory for downloaded structures
structure_path: your-path-to-example/mcp/downloads
```

#### 3. CrewAI Server Configuration

Edit `configs/crew_config.yaml` and configure:

```yaml
llm_mapper:
  your-model-name:
    base_url: http://your.llm.server:port/v1
    api_key: your-api-key
    model: openai/your-model-name
    temperature: 0

# Embedding model for RAG memory
embbeder:
  url: http://your.embedding.server:port/v1/embeddings
  model_id: BAAI/bge-m3
  api_key: your-api-key

# MCP server connection
mcp_server:
  url: http://localhost:8933/mcp
  transport: streamable-http
```

#### Required API Keys

- **Materials Project API**: Required for structure search functionality
- **LLM API**: Required for AI agent functionality (supports OpenAI-compatible APIs)
- **Embedding API**: Required for memory and RAG functionality


### Starting the Services

#### 1. Start MCP Server

Before starting the MCP server, you should setup the environment variable pointing to the POTCAR:

```bash
export PMG_VASP_PSP_DIR=/path/to/your/POTCARS/
```

Then, start the MCP server which provides tools for VASP calculations:

```bash
vaspilot_mcp --config /path/to/configs/mcp_config.yaml --port 8933
```

Or equivalently, use the provided script:
```bash
cd examples/1.Basic/mcp/
# Edit start_mcp_server.sh with your paths
./start_mcp_server.sh
```

#### 2. Start CrewAI Server

Start the main CrewAI server with `Quart` web interface:

```bash
vaspilot_quart --config /path/to/configs/crew_config.yaml --port 51293 --work-dir /path/to/work/directory --allow-path /path/to/project/ \ 
--max-concurrent-tasks 2 --max-queue-size 10
```

Or use the provided script:
```bash
cd examples/1.Basic/crew_server/
# Edit start_crew_server.sh with your paths
./start_crew_server.sh
```

### Accessing the Web Interface

Once both servers are running, access the web interface at:
```
http://localhost:51293
```

From the web interface, you can:
- Submit new calculation tasks
- Monitor running calculations
- View calculation history
- Download results and analysis reports

## Example Prompts

### Band structure and Density of States (DOS) calculations

**Example Prompt**: Calculate the band structure of 2H phase MoS2. Use IVDW=11 in relaxation.

**Workflow and Results**: 
![simple mission](figs/Fig_2.png)

### Other Missions:

1. **ENCUT Convergence Test**: Use dierent ENCUT (from 300 to 500) to calculate the total energy of 2H phase MoS2.

2. Calculate and compare the c lattice constant of 2H MoS2
with 12.294 Å (experiment value) using dierent vdW
corrections.
The settings of dierent vdW functionals are listed below:
.....
Plot an intuitve gure as result.

3. **Band Gap Comparison**: Calculate and compare the bandgap of MoS2, MoSe2, WS2 and WSe2. Plot an intuitive gure as result.

**Results**:
![complex missions](figs/Fig_3.png)

## License

This project is licensed under the LGPL v2.1. See the [LICENSE](LICENSE) file for details.

## Authors

- **Jiaxuan Liu** - liujiaxuan23@mails.ucas.ac.cn
- **Quansheng Wu** - quansheng.wu@iphy.ac.cn

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Citation

If you find VASPilot helpful, you are welcome to cite this article:

```bibtex
@misc{liu2025vaspilot,
      title={VASPilot: MCP-Facilitated Multi-Agent Intelligence for Autonomous VASP Simulations}, 
      author={Jiaxuan Liu and Tiannian Zhu and Caiyuan Ye and Zhong Fang and Hongming Weng and Quansheng Wu},
      year={2025},
      eprint={2508.07035},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2508.07035}, 
}
```

## Relevant Links

### CrewAI
  - [Website](https://www.crewai.com/)
  - [Github Repository](https://github.com/crewAIInc/crewAI)
  - [Document](https://docs.crewai.com/en/introduction)

### MCP
  - [Github Repositories](https://github.com/modelcontextprotocol)
  - [Document](https://modelcontextprotocol.io/docs/getting-started/intro)
  - [FastMCP](https://github.com/jlowin/fastmcp)

### Flask
  - [Gihub Repository](https://github.com/pallets/flask)
  - [Website](https://flask.palletsprojects.com/en/stable/)

### Quart
  - [Gihub Repository](https://github.com/pallets/quart)
  - [Website](https://quart.palletsprojects.com/en/latest/)

### Pymatgen
  - [Github Repository](https://github.com/materialsproject/pymatgen)
  - [Website](https://pymatgen.org/)