import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from .python_plot import safe_execute_plot_code
import uuid
from fastmcp import FastMCP, Context
from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints
from ase.dft.kpoints import BandPath
import yaml
import math
import numpy as np
import pickle
from .vasp_calculate import vasp_relaxation, vasp_scf, vasp_nscf, check_status, cancel_slurm_job
from .struct_tools import search_materials_project, analyze_crystal_structure, create_crystal_structure, make_supercell, rotate_structure, symmetrize_structure
from pydantic import BaseModel, Field
from .sqlite_database import VaspCalculationDB
def main(config_path: str = None, port: int = 8933, host: str = "0.0.0.0"):
    
    # 加载配置文件
    if config_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent.parent
        config_path = f"{project_root}/configs/mcp_config.yaml"
    
    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    db_path = settings['db_path']
    attachment_path = settings['attachment_path']
    mp_api_key = settings['mp_api_key']
    structure_path = settings['structure_path']

    # 初始化SQLite数据库
    db = VaspCalculationDB(db_path=db_path)

    mcp = FastMCP("VASP Agent", port=port, host=host, stateless_http=True)

    def write_record(calculation_id: str, data: dict):
        """写入计算记录到SQLite数据库"""
        db.write_record(calculation_id, data)
        return

    def read_record(calculation_id: str):
        """从SQLite数据库读取计算记录"""
        return db.read_record(calculation_id)

    def extract_calculation_data(calculation_ids: List[str]) -> Dict[str, Any]:
        """从计算记录中提取数据"""
        data = {}
        for calc_id in calculation_ids:
            record = read_record(calc_id)
            if record is not None:
                data[calc_id] = record
            else:
                data[calc_id] = None
        return data

    def extract_llm_friendly_result(data: Dict[str, Any]) -> Dict[str, Any]:
        llm_friendly_result = {}
        llm_friendly_result = {
            "status": data.get("status", "unknown"),
            "error": data.get("error"),
            "slurm_id": data.get("slurm_id"),
            "calculate_path": data.get("calculate_path"),
            "calc_type": data.get("calc_type"),
        }
        if data.get("calc_type") == "relaxation":
            llm_friendly_result.update({
                "total_energy": data.get("total_energy"),
                "max_force": data.get("max_force"),
                "stress": data.get("stress"),
                "ionic_steps": data.get("ionic_steps")
            })
        elif data.get("calc_type") == "scf":
            llm_friendly_result.update({
                "total_energy": data.get("total_energy"),
                "efermi": data.get("efermi"),
                "band_gap": data.get("band_gap"),
                "is_metal": data.get("is_metal")
            })
        elif data.get("calc_type") == "nscf":
                llm_friendly_result.update({
                    "efermi": data.get("efermi"),
                    "band_gap": data.get("band_gap"),
                    "is_metal": data.get("is_metal")
                })
        return llm_friendly_result
    @mcp.tool(name="vasp_relaxation")
    async def vasp_relaxation_tool(structure_path: str, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        提交VASP结构优化计算任务
        
        Args:
            structure_path: 结构文件路径（支持CIF、POSCAR等格式）
            incar_tags: 额外的INCAR参数字典，会与默认设置合并。除非用户指定，不要擅自修改。
            kpoint_num: K点网格设置，格式为(nx, ny, nz)的元组，如果不提供则使用自动密度40
            potcar_map: POTCAR标签设置，格式为{element: potcar}的字典。例如{"Bi": "Bi_d", "Se": "Se"}。除非用户指定，不要擅自修改。
        Returns:
            包含任务提交结果的字典，包含以下键：
            - calculation_id: 计算任务唯一标识符
            - slurm_id: SLURM任务ID
            - success: 任务提交是否成功
            - error: 错误信息（如果有）
            - status: 任务状态（pending/failed）
        """
        # 转换输入参数
        
        # 生成随机UUID
        calculation_id = str(uuid.uuid4())
        struct = Structure.from_file(structure_path)
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        incar = {}
        incar.update(settings['VASP_default_INCAR']['relaxation'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # 执行计算
        result = vasp_relaxation(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )
        
        # 保存记录
        result['calculation_id'] = calculation_id
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="vasp_scf")
    async def vasp_scf_tool(restart_id: Optional[str] = None, structure_path: Optional[str] = None, soc: bool=True, incar_tags: Optional[Dict] = None, kpoint_num: Optional[tuple[int, int, int]] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        提交VASP自洽场（SCF）计算任务
        
        Args:
            restart_id: 前序计算的ID，如果提供则使用该计算的结构和电荷密度
            structure_path: 结构文件路径，当不提供restart_id时必需
            soc: 是否包含自旋轨道耦合效应，默认为True
            incar_tags: 额外的INCAR参数字典，会与默认设置合并。除非用户指定，不要擅自修改。
            kpoint_num: K点网格设置，格式为(nx, ny, nz)的元组，如果不提供则使用自动密度40
            potcar_map: POTCAR标签设置，格式为{element: potcar}的字典。例如{"Bi": "Bi_pv", "Se": "Se_pv"}。除非用户指定，不要擅自修改。
        Returns:
            包含任务提交结果的字典，包含以下键：
            - calculation_id: 计算任务唯一标识符
            - slurm_id: SLURM任务ID
            - success: 任务提交是否成功
            - error: 错误信息（如果有）
            - status: 任务状态（pending/failed）
        """
        # 转换输入参数
        
        # 生成随机UUID
        calculation_id = str(uuid.uuid4())
        if restart_id is not None:
            restart_record = read_record(restart_id)
            if restart_record is None:
                return {"success": False, "error": f"Restart record {restart_id} not found"}
            struct = restart_record['structure']
            chgcar_path = os.path.join(restart_record['calculate_path'], "CHGCAR")
            wavecar_path = os.path.join(restart_record['calculate_path'], "WAVECAR")
        else:
            if structure_path is None:
                return {"success": False, "error": "structure_path is required when restart_id is not provided"}
            else:
                try:
                    struct = Structure.from_file(structure_path)
                except Exception as e:
                    return {"success": False, "error": f"Failed to read structure from {structure_path}: {e}"}
            chgcar_path = None
            wavecar_path = None
        if kpoint_num is None:
            factor = 40 * np.power(struct.lattice.a * struct.lattice.b * struct.lattice.c / struct.lattice.volume , 1/3)
            kpoint_float = (factor/struct.lattice.a, factor/struct.lattice.b, factor/struct.lattice.c)
            kpoint_num = (max(math.ceil(kpoint_float[0]), 1), max(math.ceil(kpoint_float[1]), 1), max(math.ceil(kpoint_float[2]), 1))
        kpts = Kpoints.gamma_automatic(kpts = kpoint_num)
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['scf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['scf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # 执行计算
        result = vasp_scf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chgcar_path,
            wavecar_path=wavecar_path,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )
        
        # 保存记录
        result['calculation_id'] = calculation_id
        result['soc'] = soc
        result['incar_tags'] = incar_tags
        result['restart_id'] = restart_id
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="vasp_nscf")
    async def vasp_nscf_tool(restart_id: str, soc: bool=True, incar_tags: Optional[Dict] = None, kpath: Optional[str] = None, n_kpoints: Optional[int] = None, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
        """
        提交VASP非自洽场（NSCF）计算任务，用于计算能带结构。提供的INCAR参数应当尽可能与前序SCF计算一致。
        
        Args:
            restart_id: 前序SCF计算的ID（必需），用于获取收敛的电荷密度和波函数
            soc: 是否包含自旋轨道耦合效应，默认为True
            incar_tags: 额外的INCAR参数字典，会与默认设置合并。除非用户指定，不要擅自修改。
            kpath: K点路径设置，可以是以下形式：
                - None: 使用pymatgen自动生成的高对称路径
                - str: 用户指定的路径，例如"GMKG"
                除非用户指定，不要擅自修改。
            n_kpoints: 每段K点路径的点数，默认为40
            potcar_map: POTCAR标签设置，格式为{element: potcar}的字典。例如{"Bi": "Bi_pv", "Se": "Se_pv"}。除非用户指定，不要擅自修改。
        Returns:
            包含任务提交结果的字典，包含以下键：
            - calculation_id: 计算任务唯一标识符
            - slurm_id: SLURM任务ID
            - success: 任务提交是否成功
            - error: 错误信息（如果有）
            - status: 任务状态（pending/failed）
        """
        # 生成随机UUID
        calculation_id = str(uuid.uuid4())
        
        # 获取结构和前序计算文件
        scf_record = read_record(restart_id)
        if scf_record is None:
            return {"success": False, "error": f"SCF record {restart_id} not found"}
        struct: Structure = scf_record['structure']
        chgcar_path = os.path.join(scf_record['calculate_path'], "CHGCAR")
        wavecar_path = os.path.join(scf_record['calculate_path'], "WAVECAR")
        
        # 设置k点路径
        from pymatgen.symmetry.bandstructure import HighSymmKpath
        kpath_obj = HighSymmKpath(struct)
        if kpath_obj.kpath is None:
            return {"success": False, "error": "Failed to generate k-path for the structure"}

        n_kpoints = 16 if n_kpoints is None else n_kpoints
        if kpath is None:
            # 使用pymatgen自动生成的高对称路径
            kpts = Kpoints.automatic_linemode(n_kpoints, kpath_obj)
        else:
            # 使用用户指定的路径
            kpts_ase: BandPath = struct.to_ase_atoms().get_cell().bandpath(kpath, npoints=n_kpoints, eps=1e-2)
            high_sym_points = []
            labels = []
            high_sym_points.append(kpts_ase.special_points[kpath[0]])
            labels.append(kpath[0])
            kpath_list = list(kpath)
            for key in kpath_list[1:-1]:
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
                high_sym_points.append(kpts_ase.special_points[key])
                labels.append(key)
            high_sym_points.append(kpts_ase.special_points[kpath[-1]])
            labels.append(kpath[-1])
            kpts = Kpoints(
                comment="User specified k-path",
                style=Kpoints.supported_modes.Line_mode,
                num_kpts=n_kpoints,
                kpts=high_sym_points,
                labels=labels,
                coord_type="Reciprocal"
            )
        
        # 设置INCAR
        incar = {}
        if soc:
            incar.update(settings['VASP_default_INCAR']['nscf_soc'])
        else:
            incar.update(settings['VASP_default_INCAR']['nscf_nsoc'])
        if incar_tags is not None:
            incar.update(incar_tags)
        
        # 执行计算
        result = vasp_nscf(
            calculation_id=calculation_id,
            work_dir=settings['work_dir'],
            struct=struct,
            kpoints=kpts,
            incar_dict=incar,
            chgcar_path=chgcar_path,
            wavecar_path=wavecar_path,
            attachment_path=attachment_path,
            potcar_map=potcar_map
        )
        
        # 保存记录
        result['calculation_id'] = calculation_id
        result['soc'] = soc
        result['incar_tags'] = incar_tags
        result['restart_id'] = restart_id
        result['kpath'] = kpath
        result['n_kpoints'] = n_kpoints
        write_record(calculation_id, result)
        
        llm_friendly_result = {
            'calculation_id': calculation_id,
            'slurm_id': result['slurm_id'],
            'success': result['success'],
            'error': result['error'],
            'status': result['status'],
            'calculate_path': result['calculate_path']
        }
        return llm_friendly_result

    @mcp.tool(name="check_calculation_status")
    async def check_calculation_status_tool(calculation_ids: List[str]) -> Dict[str, Any]:
        """
        检查计算任务状态并返回结果
        
        Args:
            calculation_ids: 要检查的计算ID列表
        
        Returns:
            包含每个计算任务状态和结果的字典，格式为：
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... 其他结果数据
                }
            }
        """
        # 从记录中获取计算信息
        calc_dict = {}
        llm_friendly_result = {}
        
        # 收集有效的计算记录
        for calc_id in calculation_ids:
            record = read_record(calc_id)
            if record is not None:
                calc_dict[calc_id] = record
        
        # 对有记录的计算检查SLURM状态
        if calc_dict:
            updated_results = check_status(calc_dict)
            # 更新记录
            for calc_id, result in updated_results.items():
                write_record(calc_id, result)
                calc_dict[calc_id] = result
        
        # 为所有计算ID构建返回结果
        for calc_id in calculation_ids:
            if calc_id in calc_dict:
                data = calc_dict[calc_id]
                llm_friendly_result[calc_id] = extract_llm_friendly_result(data)
            else:
                llm_friendly_result[calc_id] = {
                    "error": f"Calculation record {calc_id} not found",
                    "status": "error"
                }
        
        return llm_friendly_result

    @mcp.tool(name="python_plot")
    async def python_plot_tool(calculation_ids: List[str], plot_code: str, description: Optional[str] = None) -> Dict[str, Any]:
        """
        执行Python画图代码，基于指定的计算结果数据
        **注意：plot_code中不要使用plt.show()或plt.savefig()， 更不要调用plt.close()。**
        **有多个图需要绘制时，多次独立调用本工具，否则图片会被覆盖**

        Args:
            calculation_ids: 计算ID列表，用于提取计算结果数据
            plot_code: Python画图代码。数据会以data字典形式提供，键为calculation_id，值为对应的计算结果。
                    
                    可使用的库和对象：
                    - matplotlib.pyplot (as plt)
                    - numpy (as np) 
                    - pandas (as pd)
                    - pymatgen.core.Structure (as Structure)
                    - pymatgen.core.Element (as Element)
                    - pymatgen.core.Lattice (as Lattice)
                    - pymatgen.electronic_structure.bandstructure.BandStructure (as BandStructure)
                    - pymatgen.electronic_structure.dos.CompleteDos (as CompleteDos)
                    - pymatgen.io.vasp.Vasprun (as Vasprun)
                    
                    数据格式说明：
                    - data是一个字典，键为calculation_id，值为对应的计算结果
                    - 计算结果包含原始的pymatgen对象，可以直接使用pymatgen的方法进行处理和画图
                    - 常见的数据结构包括：
                        * 'structure': pymatgen.core.Structure对象
                        * 'band_structure': pymatgen.electronic_structure.bandstructure.BandStructure对象  
                        * 'dos': pymatgen.electronic_structure.dos.CompleteDos对象
                        * 'total_energy': 总能量值(float)
                        * 'efermi': 费米能级(float)
                        * 'band_gap': 字典，包含"energy", "direct", "transition"三个键，分别表示能带隙能量、是否为直接带隙、能带隙的过渡点
                        * 'stress': 应力张量(float)
                        * 'eigen_values': 本征值(numpy.ndarray)
                    
                    示例代码1，绘制能带(无特殊画图需求时参考本示例)：
                    ```python
                    # 获取第一个计算数据
                    band_calc_id = list(data.keys())[0]
                    calc_data = data[band_calc_id]
                    
                    # 绘制能带结构图
                    if 'band_structure' in calc_data:
                        bs = calc_data['band_structure']  # 这是BandStructure对象
                        plt.figure(figsize=(10, 6))
                        
                        # 使用pymatgen的内置绘图方法
                        from pymatgen.electronic_structure.plotter import BSPlotter
                        plotter = BSPlotter(bs)
                        plotter.get_plot(ylim=[-2, 2])
                        plt.axhline(y=0, color='r', linestyle='--', label='费米能级')
                        plt.ylabel('能量 - E_fermi (eV)')
                        plt.xlabel('k点')
                        plt.legend()
                    ```
                    示例代码2，绘制能带(有特殊画图需求时应当参考本示例，例如需要将两个能带画在一起时)：
                    ```python
                    # 获取第一个计算数据
                    band_calc_id = list(data.keys())[0]
                    calc_data = data[band_calc_id]
                    
                    # 绘制能带结构图
                    if 'band_structure' in calc_data:
                        bs = calc_data['band_structure']  # 这是BandStructure对象
                        # 使用pymatgen读取画图数据
                        from pymatgen.electronic_structure.plotter import BSPlotter
                        plotter = BSPlotter(bs)
                        ax = plt.gca()
                        data = plotter.bs_plot_data(bs)
                        # 绘制能带结构图
                        for spin in bs.bands:
                            ls = "-" if str(spin) == "1" else "--"
                            for dist, ene in zip(data["distances"], data["energy"][str(spin)], strict=True):
                                ax.plot(dist, ene.T, ls=ls, c="blue")

                        ax.set_xticks(data["ticks"]["distance"], data["ticks"]["label"])
                        for distance in data["ticks"]["distance"]:
                            ax.axvline(x=distance, color='black', linewidth=0.5)
                        plt.axhline(y=0, color='r', linestyle='--', label='费米能级')
                        plt.xlim(np.min(data["distances"]), np.max(data["distances"]))
                        plt.ylim(-5, 5)
                        plt.ylabel('Energy - E_fermi (eV)')
                        plt.xlabel('k points')
                    ```
                    示例代码3, 绘制态密度：
                    ```python
                    # 获取态密度计算数据
                    dos_calc_id = list(data.keys())[0]
                    dos_data = data[dos_calc_id]
                    # 绘制态密度图  
                    if 'dos' in dos_data:
                        dos = dos_data['dos']  # 这是CompleteDos对象
                        plt.figure(figsize=(8, 6))
                        # 使用pymatgen的内置绘图方法
                        from pymatgen.electronic_structure.plotter import DosPlotter
                        plotter = DosPlotter()
                        plotter.add_dos("Total DOS", dos)
                        plotter.get_plot(xlim=[-5, 5])
                    ```
            description: 可选的图表描述
        
        Returns:
            包含画图结果的字典，包含以下键：
            - success: 是否成功执行
            - error: 错误信息（如果有）
            - plot_path: 生成的图片文件路径
            - calculation_data_summary: 使用的计算数据摘要
        """
        
        # 提取计算数据
        calculation_data = extract_calculation_data(calculation_ids)
        # 检查是否有有效的计算数据
        valid_data = {k: v for k, v in calculation_data.items() if v is not None}
        if not valid_data:
            return {
                'success': False,
                'error': f'没有找到有效的计算数据。提供的计算ID: {calculation_ids}',
                'plot_path': None,
                'image_base64': None,
                'calculation_data_summary': None
            }
        
        # 创建数据摘要
        data_summary = {}
        for calc_id, data in valid_data.items():
            if data:
                summary = {
                    'calculation_id': calc_id,
                    'success': data.get('success', False),
                    'total_energy': data.get('total_energy', None),
                    'available_keys': list(data.keys())
                }
                data_summary[calc_id] = summary
        
        # 执行画图代码
        success, result, image_base64 = safe_execute_plot_code(plot_code, valid_data, settings['work_dir'])
        if success:
            return {
                'success': True,
                'error': None,
                'plot_path': result,
                'description': description
            }
        else:
            return {
                'success': False,
                'error': result,
                'plot_path': None,
                'calculation_data_summary': data_summary,
                'description': description
            }

    @mcp.tool(name="search_materials_project")
    async def search_materials_project_tool(search_criteria: Dict[str, Any], limit: int = 10) -> Dict[str, Any]:
        """
        参数:
            search_criteria: 搜索条件字典，支持以下键值：
                - material_id: str, 材料ID，例如 "mp-1234"
                - formula: str, 化学式，例如 "TiO2"
                - elements: List[str], 元素列表，例如 ["Ti", "O"]
                - exclude_elements: List[str], 排除的元素列表
                - band_gap: Tuple[float, float], 带隙范围 (min, max)，例如 (1.0, 3.0)
                - energy_above_hull: Tuple[float, float], 形成能范围 (min, max)
                - num_sites: Tuple[int, int], 原子数范围 (min, max)
                - spacegroup_number: int, 空间群编号
                - crystal_system: str, 晶系, "Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal", "Trigonal", "Hexagonal", "Cubic"中的一个
                - is_gap_direct: bool, 是否为直接带隙
            limit: 返回结果的最大数量
            
        返回:
            Dict包含搜索结果和下载状态
        """
        result = search_materials_project(api_key=mp_api_key, search_criteria=search_criteria, download_path=structure_path, limit=limit)
        return result

    @mcp.tool(name="analyze_crystal_structure")
    async def analyze_crystal_structure_tool(struct_path: str) -> Dict[str, Any]:
        """
        Args:
            struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        """
        return analyze_crystal_structure(struct_path)

    @mcp.tool(name="create_crystal_structure")
    async def create_crystal_structure_tool(positions: List[List[float]], elements: List[str], lattice_vectors: List[List[float]], cartesian: bool) -> Dict[str, Any]:
        """
        给定原子位置、元素、晶格向量，创建晶体结构，并保存到文件中。
        Args:
            positions: 原子位置，格式为[[x1, y1, z1], [x2, y2, z2], ...]
            elements: 元素列表，例如 ["Li", "F"]
            lattice_vectors: 晶格向量，格式为[[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]]
            cartesian: 是否使用笛卡尔坐标
        Returns:
            包含创建的晶体结构文件路径的字典。
        """
        return create_crystal_structure(np.array(positions), elements, np.array(lattice_vectors), cartesian, structure_path)

    @mcp.tool(name="make_supercell")
    async def make_supercell_tool(struct_path: str, supercell_matrix: List[List[int]]) -> Dict[str, Any]:
        """
        参数:
            struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        """
        return make_supercell(struct_path, supercell_matrix)

    @mcp.tool(name="symmetrize_structure")
    async def symmetrize_structure_tool(struct_path: str) -> Dict[str, Any]:
        """
        参数:
            struct_input: 结构输入，可以是文件路径或pymatgen Structure对象
        """
        return symmetrize_structure(struct_path)

    @mcp.tool(name="list_calculations")
    async def list_calculations_tool(calc_type: Optional[str] = None, status: Optional[str] = None, limit: Optional[int] = 50) -> Dict[str, Any]:
        """
        列出计算记录
        
        Args:
            calc_type: 计算类型过滤 ("relaxation", "scf", "nscf")
            status: 状态过滤 ("pending", "running", "completed", "failed", "error")
            limit: 返回记录数限制，默认50
        
        Returns:
            result_dict: 包含计算记录列表的字典
        """
        try:
            calculations = db.list_calculations(calc_type=calc_type, status=status, limit=limit)
            return {
                "success": True,
                "calculations": calculations,
                "total_count": len(calculations)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "calculations": []
            }

    @mcp.tool(name="get_database_statistics")
    async def get_database_statistics_tool() -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            数据库统计信息，包括总记录数、按类型和状态的分布等
        """
        try:
            stats = db.get_statistics()
            return {
                "success": True,
                "statistics": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }

    @mcp.tool(name="delete_calculation")
    async def delete_calculation_tool(calculation_id: str) -> Dict[str, Any]:
        """
        删除计算记录
        
        Args:
            calculation_id: 要删除的计算ID
        
        Returns:
            删除操作的结果
        """
        try:
            success = db.delete_record(calculation_id)
            if success:
                return {
                    "success": True,
                    "message": f"成功删除计算记录 {calculation_id}"
                }
            else:
                return {
                    "success": False,
                    "error": f"计算记录 {calculation_id} 不存在"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"删除计算记录时出错: {str(e)}"
            }

    @mcp.tool(name="check_files_exist")
    async def check_files_exist(file_paths: list[str]) -> Dict[str, Any]:
        """
        检查列表中的文件是否存在
        
        Args:
            file_paths: 要检查的文件路径列表
        
        Returns:
            exist_dict: 包含每个文件是否存在的字典，格式为：
            {
                "file_path1": True,
                "file_path2": True,
                ......
            }
        """

        files_exist = {}
        for path in file_paths:
            files_exist[path] = os.path.exists(path)
        return files_exist

    @mcp.tool(name="read_calc_results_from_db")
    async def read_calc_results_from_db(calc_ids: list[str]) -> Dict[str, Any]:
        """
        读取计算记录
        Args:
            calculation_ids: 要读取的计算ID列表
        
        Returns:
            包含每个计算任务状态和结果的字典，格式为：
            {
                calculation_id: {
                    "slurm_id": "12345",
                    "calc_type": "relaxation",
                    "calculate_path": "/path/to/calculation",
                    "status": "running/completed/failed/error",
                    ... 其他结果数据
                }
            }
        """
        llm_friendly_results = {}
        for calc_id in calc_ids:
            data = read_record(calc_id)
            if data is not None:
                llm_friendly_results[calc_id] = extract_llm_friendly_result(data)
            else:
                llm_friendly_results[calc_id] = {"error": "calc_id not found!"}
        return llm_friendly_results

    @mcp.tool(name="cancel_slurm_job")
    async def cancel_slurm_job_tool(calc_ids: list[str]) -> Dict[str, Any]:
        """
        取消SLURM任务
        
        Args:
            calc_ids: 要取消的计算id列表
        
        Returns:
            取消操作的结果
        """
        result_dict = {}
        for calc_id in calc_ids:
            try:
                data = read_record(calc_id)
                if data is not None:
                    if data.get("status") == "running":
                        result = cancel_slurm_job(data.get("slurm_id"))
                        data["status"] = "cancelled"
                        write_record(calc_id, data)
                        result_dict[calc_id] = result
                    else:
                        result_dict[calc_id] = {"success": True, "message": f"SLURM job {data.get('slurm_id')} is not running"}
            except Exception as e:
                result_dict[calc_id] = {"success": False, "error": str(e)}
            
        return result_dict

    mcp.run(transport="streamable-http")

if __name__ == '__main__':
    main()
