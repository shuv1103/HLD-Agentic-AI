"""
Base Node class for HLD workflow nodes
"""


import os
import time 
import psutil
import logging
from abc import ABC, abstractmethod
from typing import Any,Dict
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.base import RunnableLambda

from state.schema import HLDState


class BaseNode(ABC):
    def __init__(self,node_name:str, agent: Any=None):
        self.node_name = node_name
        self.agent = agent
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level = logging.INFO,
            format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",  
        )

       

    @abstractmethod
    def execute(self, state:HLDState) -> HLDState:
        pass


    def get_runnable(self) -> Runnable:
        def __runnable_fn(state_dict: Dict[str,Any]) -> Dict[str,Any]:
            state = HLDState(**state_dict)
            updated_state = self.execute(state)
            return updated_state.dict()
        
        return RunnableLambda(__runnable_fn)
    
    def update_state_status(self, state:HLDState,stage:str, status:str, *_,**__) -> None:
        try:
            if not hasattr(state,"stage_status"):
                state.stage_status = {}
            state.stage_status[stage] = status
            self.logger.info(f"[{self.node_name}] Stage marked as '{status}'")
        except Exception as e:
            self.logger.error(f"Failed to update stage status for {self.node_name}: {str(e)}")




    def _get_output_dir(self,state:HLDState) -> str:
        base_dir = os.path.join(os.getcwd(),"outputs",state.project_name)
        self._ensure_output_dirs(base_dir)
        return base_dir
    
    def _ensure_output_dirs(self,output_dir:str) -> None:
        os.makedirs(output_dir,exist_ok=True)

    def _get_relative_path(self, from_path:str, to_path: str) -> str:
        return os.path.relpath(to_path,start=from_path)



    def _run_with_monitoring(self,func,state:HLDState, *args, **kwargs) -> HLDState:
        start_time = time.time()
        process = psutil.Process(os.getpid())

        self.logger.info(f"Node '{self.node_name}' execution started.")



        try:
            result = func(state,*args,**kwargs)
            duration= time.time() - start_time
            memory_used = process.memory_info().rss / (1024*1024)

            self.logger.info(
                f"Node '{self.node_name}' completed successfully"
                f"in {duration:.2f}s | Memory used: {memory_used:.2f} MB"
            )

            state.performance[self.node_name] = {
                "execution_time_s": round(duration,2),
                "memory_used_mb": round(memory_used,2),
                "status": "success",
            }

            return result
        except Exception as e:
            duration = time.time() - start_time
            memory_used = process.memory_info().rss / (1024*1024)

            self.logger.error(
                f"Node '{self.node_name}' failed after {duration:.2f}s: {str(e)}"
            )

            state.errors.append({
                "node": self.node_name,
                "error": str(e),
                "time_s": round(duration,2)
            })

            state.stage[self.node_name] = "failed"

            state.performance[self.node_name] = {
                "execution_time_s": round(duration,2),
                "memory_used_mb": round(memory_used,2),
                "status": "failed"
            }

            return state
