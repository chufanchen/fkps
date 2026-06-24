from agents.bc import BCAgent
from agents.cfgrl import CFGRLAgent
from agents.dcgql import DCGQLAgent
from agents.dsrl import DSRLAgent
from agents.edp import EDPAgent
from agents.fawac import FAWACAgent
from agents.fbrac import FBRACAgent
from agents.fkps import FKPSAgent
from agents.fql import FQLAgent
from agents.grad_step import GradStepAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.iql_ddpm import IQLDDPMAgent
from agents.iql_diffusion import IQLDiffusionAgent
from agents.qam import QAMAgent
from agents.qgf import QGFAgent
from agents.robust_q import RobustQAgent
from agents.sac import SACAgent

agents = dict(
    bc=BCAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    iql_ddpm=IQLDDPMAgent,
    iql_diffusion=IQLDiffusionAgent,
    cfgrl=CFGRLAgent,
    qgf=QGFAgent,
    robust_q=RobustQAgent,
    sac=SACAgent,
    qam=QAMAgent,
    edp=EDPAgent,
    dcgql=DCGQLAgent,
    dsrl=DSRLAgent,
    fawac=FAWACAgent,
    fbrac=FBRACAgent,
    fkps=FKPSAgent,
    grad_step=GradStepAgent,
)
