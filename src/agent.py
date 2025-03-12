#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree-of-Code Agent
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Callable, List, Optional
import importlib
import os
import sys
import time
import json
import logging
import tempfile
import shutil
import subprocess
import argparse
import asyncio
import sqlite3
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import inspect
import openai  # pip install openai
from colorama import Fore, Style, init
from joblib import Memory  # pip install joblib
#import dill
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
#from pydantic import BaseModel    
from typing import Optional

import docker
from docker.errors import DockerException

# Initialize Colorama and Logging
init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Define important constants
SUCCESS_THRESHOLD = "SUCCESS"  # Used to determine if a reflection indicates success
DEFAULT_CACHE_DIR = r"..\.toc_cache"
DEFAULT_MEMORY_DB = r"..\db\memory.sqlite"  # Global SQLite database file

# -------------------------
# Configuration Loading
# -------------------------
def load_config(config_path: Optional[str] = None) -> dict:
    default_config = {
        "model": "gpt-4o-mini",
        "max_tree_depth": 10,
        "max_iterations_per_node": 2,
        "code_execution_timeout": 10,
        "temperature": 0.9,
        "max_tokens": 1024,
        "logs_dir": r"..\logs",
        "cache_dir": DEFAULT_CACHE_DIR,
        "memory_db": DEFAULT_MEMORY_DB,
        "use_sqlite_memory": True,
        "max_retries": 100,
        "thread_pool_size": 100,
        "use_docker": True,
        "memory_max_size_mb": 10000,
        # changed plugins_dir value to point to the correct folder where plugins are located:
        "plugins_dir": "plugins",
        
    }
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            default_config.update(user_config)
            logger.info(f"Configuration loaded from {config_path}")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    return default_config

# -------------------------
# Global Configuration
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not defined.")
openai.api_key = OPENAI_API_KEY

# -------------------------
# Data Structures
# -------------------------
@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    execution_time: float = 0.0
    exit_code: Optional[int] = None

    def is_successful(self) -> bool:
        return self.exit_code == 0 and not self.stderr
    
    def to_dict(self) -> dict:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "exit_code": self.exit_code,
        }

@dataclass
class TreeNode:
    thought: str
    code: str
    execution_result: Optional[ExecutionResult] = None
    reflection: str = ""
    children: list = field(default_factory=list)
    depth: int = 0
    success: bool = False
    node_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    parent: Optional["TreeNode"] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "success": self.success,
            "thought": self.thought,
            "code": self.code,
            "execution_result": {
                "stdout": self.execution_result.stdout if self.execution_result else "",
                "stderr": self.execution_result.stderr if self.execution_result else "",
                "execution_time": self.execution_result.execution_time if self.execution_result else 0.0,
                "exit_code": self.execution_result.exit_code if self.execution_result else None
            } if self.execution_result else None,
            "reflection": self.reflection,
            "children": [child.node_id for child in self.children],
            "parent": self.parent.node_id if self.parent else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict, nodes_map: dict = None) -> "TreeNode":
        if nodes_map is None:
            nodes_map = {}
        exec_result = None
        if data.get("execution_result"):
            exec_result = ExecutionResult(
                stdout=data["execution_result"].get("stdout", ""),
                stderr=data["execution_result"].get("stderr", ""),
                execution_time=data["execution_result"].get("execution_time", 0.0),
                exit_code=data["execution_result"].get("exit_code")
            )
        node = cls(
            thought=data["thought"],
            code=data["code"],
            execution_result=exec_result,
            reflection=data.get("reflection", ""),
            depth=data.get("depth", 0),
            success=data.get("success", False),
            metadata=data.get("metadata", {})
        )
        node.node_id = data["node_id"]
        nodes_map[node.node_id] = node
        return node

# -------------------------
# Docker Executor
# -------------------------

class DockerExecutor:
    def __init__(self):
        self.client = self.check_docker_installation()
        self.check_docker_daemon()

    def check_docker_installation(self):
        try:
            subprocess.run(["docker", "--version"], check=True, stdout=subprocess.PIPE)
            return docker.from_env()
        except subprocess.CalledProcessError:
            raise EnvironmentError("Docker is not installed. Please install Docker from https://docs.docker.com/get-docker/")

    def check_docker_daemon(self):
        try:
            self.client.ping()
        except DockerException:
            raise RuntimeError("Docker daemon is not running. Please start the Docker service.")

    def execute_code(self, code, image="python:3.9"):
        # Créer un fichier temporaire pour le code
        temp_dir = tempfile.mkdtemp(prefix="toc_docker_")
        code_file = os.path.join(temp_dir, "solution.py")
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            # Monter le dossier temporaire dans le conteneur et exécuter le fichier
            container = self.client.containers.run(
                image,
                command=["python", "/app/solution.py"],
                volumes={temp_dir: {"bind": "/app", "mode": "ro"}},
                remove=True,
                stdout=True,
                stderr=True
            )
            return container.decode('utf-8')
        except DockerException as e:
            raise RuntimeError(f"Error executing code in Docker: {str(e)}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

# -------------------------
# Plugin Base & Manager
# -------------------------

class Plugin(ABC):
    name: str = "base_plugin"
    version: str = "0.1.0"
    description: str = "Base plugin class"

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        
    def register_plugin(self, plugin: Plugin) -> None:
        if plugin.name in self.plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered. Overwriting.")
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
        
    def unregister_plugin(self, plugin_name: str) -> None:
        if plugin_name in self.plugins:
            self.plugins[plugin_name].shutdown()
            del self.plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            logger.warning(f"Plugin '{plugin_name}' not registered.")
            
    def register_hook(self, hook_name: str, callback: Callable) -> None:
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
        logger.debug(f"Registered hook callback for '{hook_name}'")
        
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        if hook_name not in self.hooks:
            return []
        results = []
        for callback in self.hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook '{hook_name}' callback: {e}")
        return results
        
    def load_plugin_from_path(self, plugin_path: str) -> Optional[Plugin]:
        try:
            module = importlib.import_module(plugin_path)
            logger.debug(f"Imported module: {plugin_path}")
            
            if hasattr(module, 'PLUGIN_CLASS'):
                plugin_class = module.PLUGIN_CLASS
                logger.debug(f"Found PLUGIN_CLASS: {plugin_class}")
                plugin = plugin_class()
                self.register_plugin(plugin)
                return plugin
                
            if hasattr(module, 'plugin_instance'):
                plugin = module.plugin_instance
                logger.debug(f"Found plugin_instance: {plugin}")
                self.register_plugin(plugin)
                return plugin

            plugin_classes = [
                obj for name, obj in inspect.getmembers(module)
                if inspect.isclass(obj) and issubclass(obj, Plugin) and obj != Plugin
            ]
            
            if not plugin_classes:
                plugin_classes = [
                    obj for name, obj in inspect.getmembers(module)
                    if inspect.isclass(obj) and "Plugin" in name and name != "Plugin"
                ]
                
            if not plugin_classes:
                logger.error(f"No Plugin subclass found in module {plugin_path}")
                return None
                
            plugin_class = plugin_classes[0]
            plugin = plugin_class()
            self.register_plugin(plugin)
            return plugin
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_path}: {e}")
            return None
            
    def initialize_all_plugins(self, context: Dict[str, Any]) -> None:
        logger.info("Initializing all plugins...")
        for name, plugin in self.plugins.items():
            try:
                plugin.initialize(context)
                logger.info(f"Initialized plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {name}: {e}")
                
    def shutdown_all_plugins(self) -> None:
        for name, plugin in list(self.plugins.items()):
            try:
                plugin.shutdown()
                logger.info(f"Shutdown plugin: {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown plugin {name}: {e}")

# -------------------------
# Prompt Manager
# -------------------------
class PromptManager:
    THOUGHT_TEMPLATE = (
        "Think about the best strategy to solve the following task. "
        "Break it down into clear steps (pseudo-code or step list) without explaining the solution to the user.\n\n"
        "Task: {task}\n\nContext: {context}"
    )
    
    # Template for refactoring with existing code
    THOUGHT_TEMPLATE_WITH_CODE = (
        "Think about the best strategy to refactor or modify the following code according to this task: {task}\n\n"
        "Original code:\n```python\n{existing_code}\n```\n\n"
        "Break down your approach into clear steps (pseudo-code or step list). "
        "Focus on improving the code while maintaining its core functionality.\n\n"
        "Context: {context}"
    )
    
    WORK_STRATEGY_TEMPLATE = (
        "Given the following user input or task description:\n"
        "\"{description}\"\n\n"
        "Determine the most appropriate processing strategy from the following options:\n"
        "- refactor: If the user wants to improve existing code structure, readability, or performance without changing functionality\n"
        "- document: If the user wants to generate documentation for existing code\n"
        "- create: If the user wants to create new code from scratch\n"
        "- correct: If the user wants to fix bugs or errors in existing code\n"
        "- generic: If none of the above strategies clearly apply\n\n"
        "Respond with only one word (the strategy name), without explanation or additional text."
    )

    CODE_TEMPLATE = (
        "Based on the following thought process, generate complete Python code to implement the proposed solution. "
        "Provide only the code.\n\n"
        "Thought process: {thought}"
    )
    
    # Code template for refactoring
    CODE_TEMPLATE_WITH_ORIGINAL = (
        "Based on the following thought process, refactor this original code to implement the proposed improvements. "
        "Provide only the complete refactored code.\n\n"
        "Original code:\n```python\n{original_code}\n```\n\n"
        "Thought process for refactoring: {thought}"
    )
    
    REFLECTION_TEMPLATE = (
        "Analyze the code and execution results below to determine if the solution correctly addresses the task. "
        "If errors are detected, indicate necessary corrections.\n\n"
        "Task: {task}\n\nCode:\n{code}\n\nResults:\nStdout: {stdout}\nStderr: {stderr}\n\n"
        "Provide a short comment starting with 'SUCCESS' if everything is correct, otherwise describe the errors."
    )
    
    # Reflection template for refactoring
    REFLECTION_TEMPLATE_WITH_ORIGINAL = (
        "Analyze the refactored code and execution results below to determine if it correctly addresses the task. "
        "If errors are detected, indicate necessary corrections.\n\n"
        "Task: {task}\n\n"
        "Original code:\n```python\n{original_code}\n```\n\n"
        "Refactored code:\n{code}\n\n"
        "Results:\nStdout: {stdout}\nStderr: {stderr}\n\n"
        "Provide a short comment starting with 'SUCCESS' if the refactoring is correct and improved the code, "
        "otherwise describe the issues."
    )
    
    DOCUMENTATION_TEMPLATE = (
        "Generate detailed documentation for the following Python code that solves the specified task. "
        "The documentation should include:\n"
        "- A description of the problem\n"
        "- The solution logic\n"
        "- Key modules and functions\n"
        "- Usage examples\n\n"
        "Task: {task}\n\nCode:\n{code}\n\nProvide only the documentation text."
    )
    
    REFACTOR_TEMPLATE = (
        "Refactor the following Python code to improve readability and performance.\n"
        "Description: {description}\n\nCode:\n{code}"
    )
    
    PLUGIN_DECISION_TEMPLATE = (
        "Given the following task and reflection, determine if a specialized plugin would be helpful.\n\n"
        "Task: {task}\n\n"
        "Previous Reflection: {reflection}\n\n"
        "Available plugins: {available_plugins}\n\n"
        "If a plugin would be helpful, respond with the exact plugin name (case-sensitive). "
        "Otherwise, respond with 'none'. Provide only the plugin name or 'none' without explanation."
    )
    
    SEARCH_QUERY_TEMPLATE = (
        "Based on the task and previous reflection, what specific information should we search for?\n\n"
        "Task: {task}\n\n"
        "Previous Reflection: {reflection}\n\n"
        "Respond with a concise search query (1-5 words) that would help solve the task."
    )

    @staticmethod
    def get_work_strategy_prompt(description: str) -> str:
        return PromptManager.WORK_STRATEGY_TEMPLATE.format(description=description)
    
    @staticmethod
    def get_thought_prompt(task: str, context: str = "") -> str:
        return PromptManager.THOUGHT_TEMPLATE.format(task=task, context=context)
    
    @staticmethod
    def get_thought_prompt_with_code(task: str, existing_code: str, context: str = "") -> str:
        return PromptManager.THOUGHT_TEMPLATE_WITH_CODE.format(
            task=task, 
            existing_code=existing_code,
            context=context
        )
    
    @staticmethod
    def get_code_prompt(thought: str) -> str:
        return PromptManager.CODE_TEMPLATE.format(thought=thought)
    
    @staticmethod
    def get_code_prompt_with_original(thought: str, original_code: str) -> str:
        return PromptManager.CODE_TEMPLATE_WITH_ORIGINAL.format(
            thought=thought,
            original_code=original_code
        )
    
    @staticmethod
    def get_reflection_prompt(task: str, code: str, exec_result: ExecutionResult) -> str:
        return PromptManager.REFLECTION_TEMPLATE.format(
            task=task, 
            code=code, 
            stdout=exec_result.stdout, 
            stderr=exec_result.stderr
        )
    
    @staticmethod
    def get_reflection_prompt_with_original(task: str, original_code: str, code: str, exec_result: ExecutionResult) -> str:
        return PromptManager.REFLECTION_TEMPLATE_WITH_ORIGINAL.format(
            task=task,
            original_code=original_code,
            code=code, 
            stdout=exec_result.stdout, 
            stderr=exec_result.stderr
        )
    
    @staticmethod
    def get_documentation_prompt(task: str, code: str) -> str:
        return PromptManager.DOCUMENTATION_TEMPLATE.format(task=task, code=code)
    
    @staticmethod
    def get_refactor_prompt(code: str, description: str) -> str:
        return PromptManager.REFACTOR_TEMPLATE.format(code=code, description=description)
        
    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        return hashlib.md5(prompt.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_plugin_decision_prompt(task: str, reflection: str, available_plugins: list) -> str:
        return PromptManager.PLUGIN_DECISION_TEMPLATE.format(
            task=task,
            reflection=reflection,
            available_plugins=", ".join(available_plugins) if available_plugins else "None"
        )
    
    @staticmethod
    def get_search_query_prompt(task: str, reflection: str) -> str:
        return PromptManager.SEARCH_QUERY_TEMPLATE.format(
            task=task,
            reflection=reflection
        )

# -------------------------
# Enhanced Memory Brain with SQLite
# -------------------------
class MemoryBrain:
    def __init__(self, db_path: str, max_size_mb: int = 10000):
        self.db_path = Path(db_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, timeout=10, isolation_level=None) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    additional_data TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON logs(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON logs(source)")
            conn.commit()
            logger.debug(f"Database initialized at {self.db_path}")

    def rotate_if_needed(self) -> None:
        if self.db_path.exists() and self.db_path.stat().st_size > self.max_size_bytes:
            backup = self.db_path.with_suffix(".bak")
            if backup.exists():
                backup.unlink()  # Remove the old backup.
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout = 5000;")
                backup_conn = sqlite3.connect(str(backup), timeout=10)
                backup_conn.execute("PRAGMA busy_timeout = 5000;")
                conn.backup(backup_conn)
                backup_conn.close()
                conn.execute("DELETE FROM logs")
                conn.commit()
            logger.info(f"Database rotated. A single backup copy (.bak) is saved at {backup}.")

    def append(self, content: str) -> None:
        self.rotate_if_needed()
        try:
            data = json.loads(content)
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout = 5000;")
                conn.execute(
                    "INSERT INTO logs (timestamp, source, event_type, feedback, additional_data) VALUES (?, ?, ?, ?, ?)",
                    (
                        data.get("timestamp", datetime.now().isoformat()),
                        data.get("source", "Unknown"),
                        data.get("event_type", "INFO"),
                        data.get("feedback", ""),
                        json.dumps(data.get("additional_data", {}))
                    )
                )
                conn.commit()
        except (json.JSONDecodeError, sqlite3.Error) as e:
            logger.error(f"Error appending to database: {e}")
            with open(f"{self.db_path}.fallback.log", "a", encoding="utf-8") as f:
                f.write(f"{content}\n")

    def read_all(self, limit: int = 100) -> str:
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout = 5000;")
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                logs = []
                for row in rows:
                    log_entry = {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "source": row["source"],
                        "event_type": row["event_type"],
                        "feedback": row["feedback"],
                        "additional_data": json.loads(row["additional_data"]) if row["additional_data"] else {}
                    }
                    logs.append(log_entry)
                return json.dumps(logs, indent=2)
        except sqlite3.Error as e:
            logger.error(f"Error reading from database: {e}")
            return "[]"

    def query(self, event_type: Optional[str] = None, source: Optional[str] = None, limit: int = 20) -> list:
        query = "SELECT * FROM logs WHERE 1=1"
        params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout = 5000;")
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                logs = []
                for row in rows:
                    log_entry = {
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "source": row["source"],
                        "event_type": row["event_type"],
                        "feedback": row["feedback"],
                        "additional_data": json.loads(row["additional_data"]) if row["additional_data"] else {}
                    }
                    logs.append(log_entry)
                return logs
        except sqlite3.Error as e:
            logger.error(f"Error querying database: {e}")
            return []

# -------------------------
# MetaLearner with Enhanced Feedback Management
# -------------------------
class MetaLearner:
    def __init__(self, memory: MemoryBrain):
        self.memory = memory
        self.feedback_history: list = []
        logger.debug("MetaLearner initialized")

    def add_feedback(self, feedback: str, source: str = "Unknown", event_type: str = "INFO", additional_data: Optional[dict] = None) -> None:
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "source": source,
            "event_type": event_type,
            "feedback": feedback,
            "additional_data": additional_data or {}
        }
        self.feedback_history.append(event)
        self.memory.append(json.dumps(event))
        logger.debug(f"Feedback added: {event_type} from {source}")

    def get_recent_context(self, limit: int = 5) -> str:
        return "\n".join(json.dumps(event) for event in self.feedback_history[-limit:])

    def get_relevant_context(self, event_type: Optional[str] = None, source: Optional[str] = None, limit: int = 5) -> str:
        filtered_events = self.memory.query(event_type=event_type, source=source, limit=limit)
        return json.dumps(filtered_events, indent=2)

    def adjust_prompt(self, base_prompt: str, agent_name: str) -> str:
        recent = self.get_recent_context(2)
        error_context = self.get_relevant_context(event_type="ERROR", limit=2)
        success_context = self.get_relevant_context(event_type="SUCCESS", limit=2)
        adjustment = (
            "\n\n# ADDITIONAL CONTEXT\n"
            f"Agent: {agent_name}\n"
            f"Recent feedback:\n{recent}\n"
            f"Recent errors:\n{error_context}\n"
            f"Recent successes:\n{success_context}\n"
            "\n# END CONTEXT\n"
        )
        return base_prompt + adjustment

# -------------------------
# Enhanced LLM Client with Advanced Caching
# -------------------------
class LLMClient:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1, max_retries: int = 3, cache_dir: str = DEFAULT_CACHE_DIR, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)
        self.memory = Memory(location=cache_dir, verbose=0)
        self._cached_request = self.memory.cache(self._raw_request)
        self.max_tokens = max_tokens
        logger.info(f"LLMClient initialized with model {model}, cache in {cache_dir}")

    @retry(retry=retry_if_exception_type((openai.APIError, openai.RateLimitError, openai.Timeout)),
           stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _raw_request(self, prompt: str, system_message: str = "You are a code generation assistant.") -> str:
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content.strip()
            logger.debug(f"LLM raw request successful for prompt hash: {PromptManager.compute_prompt_hash(prompt)}")
            return content
        except Exception as e:
            logger.error(f"Error in LLM call: {str(e)}")
            raise

    async def async_request(self, prompt: str, system_message: str = "You are a code generation assistant.") -> str:
        loop = asyncio.get_event_loop()
        prompt_hash = PromptManager.compute_prompt_hash(prompt)
        logger.info(f"Async LLM request with prompt hash: {prompt_hash}")
        try:
            result = await loop.run_in_executor(None, lambda: self._cached_request(prompt, system_message))
            logger.info(f"Async LLM request completed for hash: {prompt_hash}")
            return result
        except Exception as e:
            logger.error(f"Async LLM request failed: {str(e)}")
            retry_count = 0
            while retry_count < 2:
                try:
                    response = await openai.chat.completions.acreate(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content.strip()
                    logger.info(f"Direct async LLM call successful for hash: {prompt_hash}")
                    return content
                except Exception as inner_e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Error in direct async LLM call: {inner_e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            logger.error("Failed to get async response from LLM after all retries.")
            return "Error: Unable to generate content after multiple attempts."

    def request(self, prompt: str, system_message: str = "You are a code generation assistant.") -> str:
        prompt_hash = PromptManager.compute_prompt_hash(prompt)
        logger.info(f"LLM request with prompt hash: {prompt_hash}")
        try:
            content = self._cached_request(prompt, system_message)
            logger.info(f"LLM request completed (with cache) for hash: {prompt_hash}")
            return content
        except Exception as e:
            logger.error(f"Cached request failed: {str(e)}")
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content.strip()
                    logger.info(f"Direct LLM call successful for hash: {prompt_hash}")
                    return content
                except Exception as inner_e:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning(f"Error in direct LLM call: {inner_e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
            logger.error("Failed to get response from LLM after all retries.")
            return "Error: Unable to generate content after multiple attempts."

    def generate_thought(self, task: str, context: str = "") -> str:
        prompt = PromptManager.get_thought_prompt(task, context)
        thought = self.request(prompt)
        logger.info(f"Generated thought: {thought[:80]}...")
        return thought

    def generate_thought_for_refactoring(self, task: str, existing_code: str, context: str = "") -> str:
        prompt = PromptManager.get_thought_prompt_with_code(task, existing_code, context)
        thought = self.request(prompt)
        logger.info(f"Generated refactoring thought: {thought[:80]}...")
        return thought

    def generate_code(self, thought: str) -> str:
        prompt = PromptManager.get_code_prompt(thought)
        code = self.request(prompt)
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        logger.info(f"Generated code (first 80 chars): {code[:80]}...")
        return code

    def generate_code_for_refactoring(self, thought: str, original_code: str) -> str:
        prompt = PromptManager.get_code_prompt_with_original(thought, original_code)
        code = self.request(prompt)
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        logger.info(f"Generated refactored code (first 80 chars): {code[:80]}...")
        return code

    def generate_reflection(self, task: str, code: str, exec_result: ExecutionResult) -> str:
        prompt = PromptManager.get_reflection_prompt(task, code, exec_result)
        reflection = self.request(prompt)
        logger.info(f"Generated reflection: {reflection[:80]}...")
        return reflection

    def generate_reflection_for_refactoring(self, task: str, original_code: str, code: str, exec_result: ExecutionResult) -> str:
        prompt = PromptManager.get_reflection_prompt_with_original(task, original_code, code, exec_result)
        reflection = self.request(prompt)
        logger.info(f"Generated refactoring reflection: {reflection[:80]}...")
        return reflection

# -------------------------
# CodeExecutor
# -------------------------
class CodeExecutor:
    def execute_code(self, code: str, timeout: int = 10) -> ExecutionResult:
        use_docker = os.environ.get("USE_DOCKER", "true").lower() == "true"
        if use_docker:
            try:
                docker_executor = DockerExecutor()
                output = docker_executor.execute_code(code, image="python:3.9")
                return ExecutionResult(stdout=output, stderr="", execution_time=0.0, exit_code=0)
            except Exception as e:
                logger.error(f"Error during Docker code execution: {str(e)}")
                return ExecutionResult(stdout="", stderr=f"Docker execution error: {str(e)}", execution_time=0.0, exit_code=1)
        else:
            temp_dir = tempfile.mkdtemp(prefix="toc_node_")
            code_file = os.path.join(temp_dir, "solution.py")
            try:
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(code)
                if os.name != "nt":
                    os.chmod(code_file, 0o755)
                start_time = time.time()
                proc = subprocess.run([sys.executable, code_file], capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
                execution_time = time.time() - start_time
                result = ExecutionResult(stdout=proc.stdout.strip(), stderr=proc.stderr.strip(), execution_time=execution_time, exit_code=proc.returncode)
                logger.info(f"Code execution completed in {result.execution_time:.2f}s with exit code {result.exit_code}.")
                return result
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                logger.warning(f"Code execution timed out after {timeout} seconds")
                return ExecutionResult(stdout="", stderr=f"Execution timed out after {timeout} seconds", execution_time=execution_time, exit_code=124)
            except Exception as e:
                logger.error(f"Error during code execution: {str(e)}")
                return ExecutionResult(stdout="", stderr=f"Execution error: {str(e)}", execution_time=0.0, exit_code=1)
            finally:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary directory: {str(e)}")

    async def execute_code_async(self, code: str, timeout: int = 10) -> ExecutionResult:
        def _execute():
            return self.execute_code(code, timeout)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _execute)
        return result

# -------------------------
# TreeOfCodeEngine with Auto-Save, Resume and Plugin Usage Decision
# -------------------------
class TreeOfCodeEngine:
    def __init__(self, task: str, llm_client: LLMClient, meta: MetaLearner, max_depth: int, max_iterations: int, 
                 thread_pool_size: int = 4, auto_save_interval: int = 10, checkpoint_dir: Optional[Path] = None,
                 plugin_manager: Optional[PluginManager] = None):
        self.task = task
        self.llm_client = llm_client
        self.meta = meta
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.thread_pool_size = thread_pool_size
        self.root: Optional[TreeNode] = None
        self.all_leaves: list = []
        self.executor = CodeExecutor()
        self.all_nodes: dict = {}
        self.auto_save_interval = auto_save_interval  # in seconds
        self.checkpoint_dir = checkpoint_dir or (Path(".") / "checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.plugin_manager = plugin_manager
        self.original_code = None  # For storing original code in refactoring tasks
        logger.info(f"TreeOfCodeEngine initialized for task: {task[:50]}...")

    def build_tree(self) -> None:
        """Build a solution tree for code creation (from scratch)"""
        logger.info("Starting tree building for code creation")
        last_save_time = time.time()
        initial_thought = self.llm_client.generate_thought(self.task)
        self.meta.add_feedback("Initial thought generated", source="LLM", event_type="LLM_THOUGHT",
                                 additional_data={"prompt": PromptManager.get_thought_prompt(self.task), "result": initial_thought})
        initial_code = self.llm_client.generate_code(initial_thought)
        self.meta.add_feedback("Initial code generated", source="LLM", event_type="LLM_CODE",
                                 additional_data={"prompt": PromptManager.get_code_prompt(initial_thought), "result": initial_code})
        initial_exec = self.executor.execute_code(initial_code)
        initial_reflection = self.llm_client.generate_reflection(self.task, initial_code, initial_exec)
        self.meta.add_feedback("Initial reflection generated", source="LLM", event_type="LLM_REFLECTION",
                                 additional_data={"prompt": PromptManager.get_reflection_prompt(self.task, initial_code, initial_exec), "result": initial_reflection})
        success = initial_reflection.upper().startswith(SUCCESS_THRESHOLD)
        self.root = TreeNode(thought=initial_thought, code=initial_code, execution_result=initial_exec, reflection=initial_reflection, depth=0, success=success)
        self.all_nodes[self.root.node_id] = self.root
        queue = deque([self.root])
        
        # Process the tree, generating child nodes
        self._process_tree_nodes(queue, last_save_time)
        
    def build_tree_for_refactoring(self, existing_code: str) -> None:
        """Build a solution tree for refactoring existing code"""
        logger.info("Starting tree building for refactoring")
        last_save_time = time.time()
        
        # Store the original code for future reference
        self.original_code = existing_code
        
        # Generate initial thought based on existing code
        initial_thought = self.llm_client.generate_thought_for_refactoring(self.task, existing_code)
        self.meta.add_feedback("Initial refactoring thought generated", source="LLM", event_type="LLM_THOUGHT",
                               additional_data={"prompt": PromptManager.get_thought_prompt_with_code(self.task, existing_code), 
                                               "result": initial_thought})
        
        # Generate initial refactored code
        initial_code = self.llm_client.generate_code_for_refactoring(initial_thought, existing_code)
        self.meta.add_feedback("Initial refactored code generated", source="LLM", event_type="LLM_CODE",
                               additional_data={"prompt": PromptManager.get_code_prompt_with_original(initial_thought, existing_code), 
                                               "result": initial_code})
        
        # Execute and reflect on the refactored code
        initial_exec = self.executor.execute_code(initial_code)
        initial_reflection = self.llm_client.generate_reflection_for_refactoring(
            self.task, existing_code, initial_code, initial_exec
        )
        self.meta.add_feedback("Initial refactoring reflection generated", source="LLM", event_type="LLM_REFLECTION",
                               additional_data={"prompt": PromptManager.get_reflection_prompt_with_original(
                                   self.task, existing_code, initial_code, initial_exec), 
                                   "result": initial_reflection})
        
        success = initial_reflection.upper().startswith(SUCCESS_THRESHOLD)
        
        # Create the root node with the initial refactoring
        self.root = TreeNode(
            thought=initial_thought, 
            code=initial_code, 
            execution_result=initial_exec, 
            reflection=initial_reflection, 
            depth=0, 
            success=success,
            metadata={"original_code": existing_code, "is_refactoring": True}
        )
        self.all_nodes[self.root.node_id] = self.root
        queue = deque([self.root])
        
        # Process the tree, generating child nodes
        self._process_tree_nodes(queue, last_save_time)
    
    def _process_tree_nodes(self, queue: deque, last_save_time: float) -> None:
        """
        Process nodes in the tree queue, generating children for each node.
        This is a common method used by both build_tree and build_tree_for_refactoring.
        """
        while queue:
            current = queue.popleft()
            
            # Auto-save checkpoint if needed
            if time.time() - last_save_time > self.auto_save_interval:
                autosave_path = str(self.checkpoint_dir / (self.root.node_id + "_autosave.json"))
                self.save_tree(autosave_path)
                logger.info(f"Auto-save performed at {autosave_path}")
                last_save_time = time.time()
                
            # If node is successful or at max depth, add to leaves
            if current.success or current.depth >= self.max_depth:
                self.all_leaves.append(current)
            else:
                # Generate child nodes in parallel
                with ThreadPoolExecutor(max_workers=min(self.thread_pool_size, self.max_iterations)) as executor:
                    futures = [executor.submit(self._generate_child_node_sync, current) for _ in range(self.max_iterations)]
                    children = [future.result() for future in as_completed(futures)]
                    
                current.children.extend(children)
                queue.extend(children)
                for node in children:
                    self.all_nodes[node.node_id] = node
                    
        # Trigger hooks after tree is built
        if self.plugin_manager:
            self.plugin_manager.trigger_hook("after_tree_build", engine=self)
            
        logger.info(f"Tree building completed with {len(self.all_nodes)} nodes")

    def _generate_child_node_sync(self, parent_node: TreeNode) -> TreeNode:
        # Check if this is a refactoring task
        is_refactoring = parent_node.metadata.get("is_refactoring", False)
        original_code = parent_node.metadata.get("original_code") or self.original_code
        
        # 1. Plugin decision
        plugin_decision_prompt = PromptManager.get_plugin_decision_prompt(
            task=self.task,
            reflection=parent_node.reflection,
            available_plugins=list(self.plugin_manager.plugins.keys()) if self.plugin_manager else []
        )
        
        plugin_choice = self.llm_client.request(plugin_decision_prompt).strip().lower()
        used_plugin_data = None
        logger.info(f"Plugin decision made: {plugin_choice}")
        # 2. Plugin usage if needed
        if plugin_choice != "none" and self.plugin_manager and plugin_choice in self.plugin_manager.plugins:
            logger.info(f"Plugin '{plugin_choice}' selected for node generation.")
            plugin = self.plugin_manager.plugins[plugin_choice]
            
            search_context_prompt = PromptManager.get_search_query_prompt(task=self.task, reflection=parent_node.reflection)
            search_query = self.llm_client.request(search_context_prompt).strip()
            logger.info(f"Generated search query: {search_query}")
            
            try:
                if hasattr(plugin, 'search') and callable(plugin.search):
                    search_results = plugin.search(search_query)
                    if search_results:
                        used_plugin_data = {
                            "plugin_name": plugin_choice,
                            "search_query": search_query,
                            "results": search_results[:5]
                        }
                        logger.info(f"Successfully used plugin '{plugin_choice}' with query '{search_query}'")
                        self.plugin_manager.trigger_hook("before_generate_child_plugin", 
                                                        parent=parent_node, 
                                                        plugin=plugin_choice,
                                                        query=search_query,
                                                        results=search_results)
                else:
                    self.plugin_manager.trigger_hook("before_generate_child_plugin", parent=parent_node, plugin=plugin_choice)
            except Exception as e:
                logger.error(f"Error using plugin '{plugin_choice}': {e}")
        
        if self.plugin_manager:
            self.plugin_manager.trigger_hook("before_generate_child", parent=parent_node, task=self.task)
        
        # Prepare context with plugin data if available
        context_with_plugin = parent_node.reflection
        if used_plugin_data:
            plugin_results_text = "\n".join([
                f"- {i+1}. {str(result)[:200]}..." if len(str(result)) > 200 else f"- {i+1}. {result}"
                for i, result in enumerate(used_plugin_data["results"])
            ])
            context_with_plugin = (
                f"{parent_node.reflection}\n\n"
                f"PLUGIN DATA FROM '{used_plugin_data['plugin_name']}':\n"
                f"Search query: {used_plugin_data['search_query']}\n"
                f"Results:\n{plugin_results_text}\n\n"
                f"Use this plugin data to improve your solution."
            )
        
        # Generate thought, code, and reflection based on task type (refactoring or creation)
        if is_refactoring:
            # For refactoring tasks
            thought_prompt = PromptManager.get_thought_prompt_with_code(self.task, original_code, context_with_plugin)
            new_thought = self.llm_client.request(thought_prompt)
            self.meta.add_feedback("Child refactoring thought generated", source="LLM", event_type="LLM_THOUGHT",
                                  additional_data={"prompt": thought_prompt, "result": new_thought})
            
            code_prompt = PromptManager.get_code_prompt_with_original(new_thought, original_code)
            new_code = self.llm_client.request(code_prompt)
            
            # Clean up code formatting if needed
            if new_code.startswith("```python"):
                new_code = new_code[len("```python"):].strip()
            elif new_code.startswith("```"):
                new_code = new_code[3:].strip()
            if new_code.endswith("```"):
                new_code = new_code[:-3].strip()
                
            self.meta.add_feedback("Child refactored code generated", source="LLM", event_type="LLM_CODE",
                                  additional_data={"prompt": code_prompt, "result": new_code})
            
            new_exec = self.executor.execute_code(new_code)
            
            reflection_prompt = PromptManager.get_reflection_prompt_with_original(
                self.task, original_code, new_code, new_exec
            )
            new_reflection = self.llm_client.request(reflection_prompt)
            self.meta.add_feedback("Child refactoring reflection generated", source="LLM", event_type="LLM_REFLECTION",
                                  additional_data={"prompt": reflection_prompt, "result": new_reflection})
        else:
            # For creation tasks (standard approach)
            thought_prompt = PromptManager.get_thought_prompt(self.task, context_with_plugin)
            new_thought = self.llm_client.request(thought_prompt)
            self.meta.add_feedback("Child thought generated", source="LLM", event_type="LLM_THOUGHT",
                                  additional_data={"prompt": thought_prompt, "result": new_thought})
            
            code_prompt = PromptManager.get_code_prompt(new_thought)
            new_code = self.llm_client.request(code_prompt)
            
            # Clean up code formatting if needed
            if new_code.startswith("```python"):
                new_code = new_code[len("```python"):].strip()
            elif new_code.startswith("```"):
                new_code = new_code[3:].strip()
            if new_code.endswith("```"):
                new_code = new_code[:-3].strip()
                
            self.meta.add_feedback("Child code generated", source="LLM", event_type="LLM_CODE",
                                  additional_data={"prompt": code_prompt, "result": new_code})
            
            new_exec = self.executor.execute_code(new_code)
            
            reflection_prompt = PromptManager.get_reflection_prompt(self.task, new_code, new_exec)
            new_reflection = self.llm_client.request(reflection_prompt)
            self.meta.add_feedback("Child reflection generated", source="LLM", event_type="LLM_REFLECTION",
                                  additional_data={"prompt": reflection_prompt, "result": new_reflection})
        
        # Determine if this solution is successful
        child_success = new_reflection.upper().startswith(SUCCESS_THRESHOLD)
        
        # Create child node with all the gathered information
        child = TreeNode(
            thought=new_thought, 
            code=new_code, 
            execution_result=new_exec, 
            reflection=new_reflection,
            depth=parent_node.depth + 1, 
            success=child_success, 
            parent=parent_node,
            metadata={
                "is_refactoring": is_refactoring,
                "original_code": original_code if is_refactoring else None,
                "plugin_used": plugin_choice if plugin_choice != "none" else None, 
                "plugin_data": used_plugin_data
            }
        )
        
        if self.plugin_manager:
            self.plugin_manager.trigger_hook("after_generate_child", child=child, parent=parent_node)
        
        return child

    def save_tree(self, filepath: str) -> None:
        if not self.root:
            logger.warning("No tree to save")
            return
        nodes_dict = {node_id: node.to_dict() for node_id, node in self.all_nodes.items()}
        tree_data = {
            "task": self.task,
            "root_id": self.root.node_id,
            "nodes": nodes_dict,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "timestamp": datetime.now().isoformat(),
            "original_code": self.original_code
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tree_data, f, indent=2)
            logger.info(f"Tree saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving tree to {filepath}: {str(e)}")
            alt_filepath = f"{filepath}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            try:
                with open(alt_filepath, 'w', encoding='utf-8') as f:
                    json.dump(tree_data, f, indent=2)
                logger.info(f"Tree saved to alternative location: {alt_filepath}")
            except Exception as e2:
                logger.error(f"Failed to save tree to alternative location: {str(e2)}")

    def get_final_solution(self) -> Optional[TreeNode]:
        success_nodes = [node for node in self.all_leaves if node.success]
        if success_nodes:
            best_solution = min(success_nodes, key=lambda node: node.depth)
            logger.info(f"Found successful solution at depth {best_solution.depth}")
            return best_solution
        logger.warning("No successful solution found")
        return None

    @classmethod
    def load_tree(cls, filepath: str, llm_client: LLMClient, meta: MetaLearner, checkpoint_dir: Optional[Path] = None, plugin_manager: Optional[PluginManager] = None) -> "TreeOfCodeEngine":
        logger.info(f"Loading tree from {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)
            engine = cls(task=tree_data["task"], llm_client=llm_client, meta=meta, max_depth=tree_data["max_depth"], 
                         max_iterations=tree_data["max_iterations"], checkpoint_dir=checkpoint_dir, plugin_manager=plugin_manager)
            nodes_map = {}
            for node_id, node_data in tree_data["nodes"].items():
                node = TreeNode.from_dict(node_data)
                nodes_map[node_id] = node
            for node_id, node_data in tree_data["nodes"].items():
                node = nodes_map[node_id]
                if node_data.get("parent"):
                    node.parent = nodes_map[node_data["parent"]]
                for child_id in node_data.get("children", []):
                    if child_id in nodes_map:
                        node.children.append(nodes_map[child_id])
            engine.root = nodes_map[tree_data["root_id"]]
            engine.all_nodes = nodes_map
            engine.all_leaves = [node for node in nodes_map.values() if not node.children or node.success or node.depth >= engine.max_depth]
            engine.original_code = tree_data.get("original_code")
            logger.info(f"Tree loaded from {filepath} with {len(engine.all_nodes)} nodes")
            return engine
        except Exception as e:
            logger.error(f"Error loading tree from {filepath}: {str(e)}")
            raise ValueError(f"Failed to load tree: {str(e)}")

    @classmethod
    def resume_checkpoint(cls, checkpoint_dir: Path, llm_client: LLMClient, meta: MetaLearner, plugin_manager: Optional[PluginManager] = None) -> Optional["TreeOfCodeEngine"]:
        checkpoints = list(checkpoint_dir.glob("*_autosave.json"))
        if checkpoints:
            latest = max(checkpoints, key=lambda f: f.stat().st_mtime)
            try:
                engine = cls.load_tree(str(latest), llm_client, meta, checkpoint_dir=checkpoint_dir, plugin_manager=plugin_manager)
                logger.info(f"Resumed tree from checkpoint: {latest}")
                return engine
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint {latest}: {e}")
                return None
        else:
            logger.info("No checkpoint files found.")
            return None

# -------------------------
# Documenter
# -------------------------
class Documenter:
    def __init__(self, llm_client: LLMClient, meta: MetaLearner):
        self.llm_client = llm_client
        self.meta = meta
        logger.debug("Documenter initialized")

    def generate_documentation(self, code: str, task: str) -> str:
        if not code:
            return "No code to document."
        prompt = PromptManager.get_documentation_prompt(task, code)
        try:
            documentation = self.llm_client.request(prompt)
            self.meta.add_feedback("Documentation generated", source="Documenter", event_type="DOCUMENTATION", additional_data={"task": task})
            doc_context = self.meta.get_recent_context(3)
            full_documentation = (f"# Documentation for '{task}'\n\n{documentation}\n\n## Recent Context\n{doc_context}\n\n"
                                  "(Documentation automatically generated by TreeOfCode Documenter)")
            logger.info(f"Documentation generated for task: {task[:50]}...")
            return full_documentation
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return (f"# Documentation for '{task}'\n\n## Overview\nThis code was generated to solve the following task: {task}\n\n"
                    "## Code\n```\n{code}\n```\n\n(Basic documentation generated due to an error in the documentation generator)")

# -------------------------
# Tree Visualizer
# -------------------------
class TreeVisualizer:
    @staticmethod
    def generate_ascii_tree(root: TreeNode, max_depth: int = None) -> str:
        if not root:
            return "Empty tree"
        lines = []
        TreeVisualizer._generate_ascii_tree_recursive(root, prefix="", is_last=True, lines=lines, max_depth=max_depth)
        return "\n".join(lines)
    
    @staticmethod
    def _generate_ascii_tree_recursive(node: TreeNode, prefix: str, is_last: bool, lines: list, max_depth: Optional[int] = None) -> None:
        if max_depth is not None and node.depth > max_depth:
            return
        connector = "└── " if is_last else "├── "
        status = "✓" if node.success else "✗"
        node_line = f"{prefix}{connector}[{node.depth}][{status}] Node {node.node_id[-8:]}"
        if node.execution_result:
            exec_status = "⚠️ Error" if node.execution_result.stderr else "✓ OK"
            exec_time = f"{node.execution_result.execution_time:.2f}s"
            node_line += f" ({exec_status}, {exec_time})"
        lines.append(node_line)
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            TreeVisualizer._generate_ascii_tree_recursive(child, prefix=new_prefix, is_last=is_last_child, lines=lines, max_depth=max_depth)

    @staticmethod
    def generate_mermaid_diagram(root: TreeNode, max_depth: int = None) -> str:
        if not root:
            return "graph TD\n    EmptyTree[Empty Tree]"
        lines = ["graph TD"]
        visited = set()
        TreeVisualizer._generate_mermaid_diagram_recursive(root, lines, visited, max_depth=max_depth)
        return "\n".join(lines)
    
    @staticmethod
    def _generate_mermaid_diagram_recursive(node: TreeNode, lines: list, visited: set, max_depth: Optional[int] = None) -> None:
        if node.node_id in visited:
            return
        if max_depth is not None and node.depth > max_depth:
            return
        visited.add(node.node_id)
        node_id = f"N{node.node_id}"
        status = "SUCCESS" if node.success else "FAILURE"
        lines.append(f"    {node_id}[\"Node {node.node_id[-8:]} ({status})\"]")
        if node.success:
            lines.append(f"    style {node_id} fill:#a3f9a3,stroke:#0c8a0c")
        else:
            lines.append(f"    style {node_id} fill:#f9a3a3,stroke:#8a0c0c")
        for child in node.children:
            child_id = f"N{child.node_id}"
            lines.append(f"    {node_id} --> {child_id}")
            TreeVisualizer._generate_mermaid_diagram_recursive(child, lines, visited, max_depth=max_depth)
    
    @staticmethod
    def generate_html_visualization(root: TreeNode, title: str = "Tree-of-Code Visualization") -> str:
        if not root:
            return "<html><body><h1>Empty Tree</h1></body></html>"
        mermaid_diagram = TreeVisualizer.generate_mermaid_diagram(root)
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .mermaid {{ margin: 20px 0; }}
                .node-details {{ margin-top: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <div class="mermaid">
            {mermaid_diagram}
            </div>
            <div class="node-details">
                <h2>Node Details</h2>
                {TreeVisualizer._generate_node_details_html(root)}
            </div>
            <script>
                mermaid.initialize({{ startOnLoad: true }});
            </script>
        </body>
        </html>
        """
        return html
    
    @staticmethod
    def _generate_node_details_html(node: TreeNode) -> str:
        status_class = "success" if node.success else "failure"
        status_text = "SUCCESS" if node.success else "FAILURE"
        exec_details = ""
        if node.execution_result:
            exec_details = f"""
            <h4>Execution Results:</h4>
            <p>Exit Code: {node.execution_result.exit_code}</p>
            <p>Execution Time: {node.execution_result.execution_time:.2f}s</p>
            """
            if node.execution_result.stdout:
                exec_details += f"""
                <h5>Standard Output:</h5>
                <pre>{node.execution_result.stdout}</pre>
                """
            if node.execution_result.stderr:
                exec_details += f"""
                <h5>Standard Error:</h5>
                <pre>{node.execution_result.stderr}</pre>
                """
        
        is_refactoring = node.metadata.get("is_refactoring", False)
        original_code = node.metadata.get("original_code", "")
        
        original_code_section = ""
        if is_refactoring and original_code:
            original_code_section = f"""
            <h4>Original Code:</h4>
            <pre>{original_code}</pre>
            """
        
        html = f"""
        <div id="node-{node.node_id}" class="node">
            <h3>Node {node.node_id} (Depth: {node.depth}, Status: <span class="{status_class}">{status_text}</span>)</h3>
            <h4>Thought:</h4>
            <pre>{node.thought}</pre>
            {original_code_section}
            <h4>Code:</h4>
            <pre>{node.code}</pre>
            {exec_details}
            <h4>Reflection:</h4>
            <pre>{node.reflection}</pre>
        </div>
        """
        for child in node.children:
            html += TreeVisualizer._generate_node_details_html(child)
        return html

# -------------------------
# Enhanced Work Processor with Error Recovery
# -------------------------
class WorkProcessor:
    def __init__(self, llm_client: LLMClient, meta: MetaLearner, session_dir: Path):
        self.llm_client = llm_client
        self.meta = meta
        self.session_dir = session_dir
        logger.info(f"WorkProcessor initialized with session directory: {session_dir}")

    def process_work(self, work: dict) -> bool:
        target = work.get("target", "")
        description = work.get("description", "")
        strategy_prompt = PromptManager.get_work_strategy_prompt(description)
        strategy = self.llm_client.request(strategy_prompt).strip().lower()
        self.meta.add_feedback(f"Work strategy deduced as '{strategy}' for target '{target}'", source="WorkProcessor", event_type="LLM_STRATEGY", additional_data={"strategy": strategy, "description": description})
        processing_method = getattr(self, f"_process_{strategy}", self._process_generic)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = self.session_dir / f"work_{strategy}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            success = processing_method(target, description, output_dir, self.llm_client)
        except Exception as e:
            self.meta.add_feedback(f"Error processing work with strategy '{strategy}' on '{target}': {str(e)}", source="WorkProcessor", event_type="ERROR", additional_data={"error": str(e), "traceback": traceback.format_exc()})
            logger.error(f"Error processing work with strategy '{strategy}' on '{target}': {str(e)}")
            success = False
        self._create_work_summary(work, output_dir, success)
        return success

    def _process_generic(self, target: str, description: str, output_dir: Path, llm_client: LLMClient) -> bool:
        generic_prompt = (
            "Given the following task description:\n"
            f"\"{description}\"\n\n"
            "Provide a detailed step-by-step plan to process this work task. Then, output the final code result."
        )
        result = llm_client.request(generic_prompt)
        output_file = output_dir / "generic_solution.py"
        output_file.write_text(result, encoding="utf-8")
        return bool(result.strip())

    def _create_work_summary(self, work: dict, output_dir: Path, success: bool) -> None:
        summary = {
            "work_type": work.get("work_type", ""),
            "target": work.get("target", ""),
            "description": work.get("description", ""),
            "config_override": work.get("config_override", {}),
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "output_directory": str(output_dir)
        }
        summary_file = output_dir / "work_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def process_work_json(self, json_path: str) -> dict:
        if not os.path.isfile(json_path):
            logger.error(f"Work JSON file not found: {json_path}")
            return {"success": False, "error": f"File not found: {json_path}", "results": []}
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                work_data = json.load(f)
            works = work_data.get("works", [])
            if not works:
                logger.warning("No works found in the JSON file.")
                return {"success": False, "error": "No works found in the JSON file", "results": []}
            results = []
            total_works = len(works)
            successful_works = 0
            for i, work in enumerate(works):
                logger.info(f"Processing work {i+1}/{total_works}: {work.get('work_type', 'unknown')} on {work.get('target', 'unknown')}")
                success = self.process_work(work)
                if success:
                    successful_works += 1
                results.append({"work_type": work.get("work_type", ""), "target": work.get("target", ""), "success": success})
            summary = {"success": successful_works > 0, "total": total_works, "successful": successful_works, "failed": total_works - successful_works, "results": results}
            summary_file = self.session_dir / "work_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Work processing completed. {successful_works}/{total_works} successful.")
            return summary
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {json_path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "results": []}
        except Exception as e:
            error_msg = f"Error processing works from {json_path}: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "results": []}

# -------------------------
# Unified Agent Orchestrator with Dynamic Plugin Integration and Checkpoint Resume
# -------------------------
class UnifiedAgentOrchestrator:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        base_logs_dir = Path(self.config.get("logs_dir", "logs"))
        base_logs_dir.mkdir(parents=True, exist_ok=True)
        session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = base_logs_dir / f"work_{session_timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        memory_db = Path(self.config.get("memory_db", DEFAULT_MEMORY_DB))
        self.memory = MemoryBrain(str(memory_db), max_size_mb=self.config.get("memory_max_size_mb", 1000))
        self.meta = MetaLearner(self.memory)
        cache_dir = self.session_dir / self.config.get("cache_dir", DEFAULT_CACHE_DIR)
        self.llm_client = LLMClient(model=self.config.get("model", "gpt-4o-mini"), temperature=self.config.get("temperature", 0.1),
                                    max_retries=self.config.get("max_retries", 3), cache_dir=str(cache_dir), max_tokens=self.config.get("max_tokens", 1024))
        self.documenter = Documenter(self.llm_client, self.meta)
        self.visualizer = TreeVisualizer()
        self.max_tree_depth = self.config.get("max_tree_depth", 3)
        self.max_iterations = self.config.get("max_iterations_per_node", 2)
        self.code_execution_timeout = self.config.get("code_execution_timeout", 10)
        self.thread_pool_size = self.config.get("thread_pool_size", 4)
        config_file = self.session_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)
        # Load plugins automatically from plugins_dir if exists.
        self.plugin_manager = PluginManager()
        plugins_dir = self.config.get("plugins_dir", "plugins")
        logger.info(f"Loading plugins from directory: {plugins_dir}")
        
        if os.path.isdir(plugins_dir):
            plugins_abs_dir = os.path.abspath(plugins_dir)
            if plugins_abs_dir not in sys.path:
                sys.path.insert(0, plugins_abs_dir)
            plugins_dir_pkg = plugins_dir.replace(os.sep, ".").replace("/", ".")
            for plugin_file in os.listdir(plugins_dir):
                if plugin_file.endswith(".py") and plugin_file != "__init__.py":
                    logger.info(f"Loading plugin from file: {plugin_file}")
                    plugin_module_name = plugin_file[:-3]
                    plugin_path_full = f"{plugins_dir_pkg}.{plugin_module_name}"
                    logger.info(f"Loading plugin from path: {plugin_path_full}")
                    plugin = self.plugin_manager.load_plugin_from_path(plugin_path_full)
                    if plugin:
                        logger.info(f"Successfully loaded plugin: {plugin.name} v{plugin.version}")
                    else:
                        logger.error(f"Failed to load plugin from {plugin_path_full}")
        
        context = {
            "plugin_manager": self.plugin_manager,
            "llm_client": self.llm_client,
            "meta_learner": self.meta,
            "config": self.config,
            "session_dir": str(self.session_dir)
        }
        self.plugin_manager.initialize_all_plugins(context)
        
        logger.info(f"UnifiedAgentOrchestrator initialized with session directory: {self.session_dir}")
        self.meta.add_feedback("Orchestrator initialized", source="System", event_type="INIT", additional_data={"config": self.config})

    def detect_strategy(self, task_description: str) -> str:
        """
        Detect the processing strategy based on the task description.
        
        Args:
            task_description: The user's task description
            
        Returns:
            str: The detected strategy (refactor, document, correct, create, or generic)
        """
        strategy_prompt = PromptManager.get_work_strategy_prompt(task_description)
        strategy = self.llm_client.request(strategy_prompt).strip().lower()
        logger.info(f"Strategy detected: {strategy} for task: {task_description[:50]}...")
        self.meta.add_feedback(f"Strategy detected: {strategy}", source="Orchestrator", event_type="STRATEGY_DETECTION")
        return strategy

    def process_task(self, task: str) -> dict:
        """
        Process a task based on its description. This is the main API entry point.
        Uses TreeOfCode for all strategies (creation, refactoring, etc.)
        
        Args:
            task: The user's task description
            
        Returns:
            dict: Results of the task processing
        """
        self.meta.add_feedback(f"Task received: {task}", source="User", event_type="TASK")
        
        # Detect the processing strategy
        strategy = self.detect_strategy(task)
        
        # For file-based strategies (refactor, document, correct), extract file path
        if strategy in ["refactor", "document", "correct"]:
            file_path = self._extract_file_path(task)
            
            if not file_path:
                # No valid file path found, return error
                error_msg = f"Could not extract a valid file path for {strategy} from the task description"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "strategy": strategy
                }
            
            # Read the file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_code = f.read()
            except Exception as e:
                error_msg = f"Error reading file {file_path}: {e}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "strategy": strategy
                }
        else:
            # For creation/generic, no existing code
            existing_code = None
            file_path = None
        
        # Set up the TreeOfCodeEngine
        checkpoint_dir = self.session_dir / "checkpoints"
        engine = TreeOfCodeEngine(task, self.llm_client, self.meta, 
                                self.max_tree_depth, self.max_iterations,
                                self.thread_pool_size, checkpoint_dir=checkpoint_dir,
                                plugin_manager=self.plugin_manager)
        
        # Use the appropriate tree building method based on strategy
        if strategy in ["refactor", "document", "correct"] and existing_code:
            # For refactoring and related tasks, use existing code
            engine.build_tree_for_refactoring(existing_code)
        else:
            # For creation and generic tasks, create from scratch
            engine.build_tree()
            
        if self.plugin_manager:
            self.plugin_manager.trigger_hook("after_tree_build", engine=engine)
            
        # Get the final solution
        solution = engine.get_final_solution()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate output files
        tree_file = self.session_dir / f"tree_{timestamp}.json"
        engine.save_tree(str(tree_file))
        
        # Determine output file name based on strategy
        if file_path and strategy in ["refactor", "document", "correct"]:
            base_name = os.path.basename(file_path)
            solution_file = self.session_dir / f"{base_name}_{strategy}_{timestamp}.py"
        else:
            solution_file = self.session_dir / f"solution_{timestamp}.py"
            
        solution_file.write_text(solution.code, encoding="utf-8")
        
        # Generate documentation
        doc_task = task
        if existing_code and strategy in ["refactor", "document", "correct"]:
            doc_task = f"{strategy.capitalize()} of {file_path}. Original code:\n```python\n{existing_code}\n```\nTask: {task}"
            
        documentation = self.documenter.generate_documentation(solution.code, doc_task)
        doc_file = self.session_dir / f"documentation_{timestamp}.md"
        doc_file.write_text(documentation, encoding="utf-8")
        
        # Generate visualization
        title = f"Tree-of-Code for {strategy}: {task[:50]}..."
        html_viz = self.visualizer.generate_html_visualization(engine.root, title=title)
        html_file = self.session_dir / f"visualization_{timestamp}.html"
        html_file.write_text(html_viz, encoding="utf-8")
        
        logger.info(f"Work files saved in: {self.session_dir}")
        
        # Prepare the result dictionary
        result = {
            "solution": solution,
            "work_path": str(self.session_dir),
            "tree_file": str(tree_file),
            "solution_file": str(solution_file),
            "doc_file": str(doc_file),
            "html_file": str(html_file),
            "strategy": strategy,
            "success": solution.success
        }
        
        # Add additional info for refactoring
        if file_path and strategy in ["refactor", "document", "correct"]:
            result["original_file"] = file_path
            
        return result
    
    def _extract_file_path(self, task_description: str) -> str:
        """
        Extract a file path from a task description using LLM.
        
        Args:
            task_description: The user's task description
            
        Returns:
            str: The extracted file path or empty string if none found
        """
        extract_prompt = (
            f"From the following task description, extract the file path that needs to be processed.\n"
            f"Task: {task_description}\n\n"
            f"Return only the file path, without any additional text. "
            f"If there's no clear file path, return 'NONE'."
        )
        
        file_path = self.llm_client.request(extract_prompt).strip()
        
        if file_path.upper() == "NONE":
            return ""
            
        # Clean up the extracted path
        file_path = file_path.strip("'\"")
        
        # Check if the file exists
        if os.path.isfile(file_path):
            return file_path
        else:
            logger.warning(f"Extracted file path does not exist: {file_path}")
            return ""
    
    def run_sync(self, resume=False):
        """
        Run the agent in synchronous mode via CLI.
        Uses the same core processing as API mode.
        
        Args:
            resume: Whether to resume from a checkpoint
        """
        try:
            print(Fore.YELLOW + "=== ENHANCED AUTONOMOUS AI AGENT - UNIFIED MODE (Tree-of-Code) ===" + Style.RESET_ALL)
            print(f"Session directory: {self.session_dir}")
            user_input = input("How can I help you today?\n>>> ").strip()
            
            if not user_input:
                raise ValueError("No task provided")
            
            # Detect strategy
            strategy = self.detect_strategy(user_input)
            print(Fore.CYAN + f"Detected strategy: {strategy}" + Style.RESET_ALL)
            
            # File-based strategies need a file path
            if strategy in ["refactor", "document", "correct"]:
                if self._extract_file_path(user_input):
                    file_path = self._extract_file_path(user_input)
                else:
                    file_path = input("Please provide the path to the file:\n>>> ").strip()
                    # Update the task with the file path
                    user_input = f"{user_input} File: {file_path}"
            
            # Process the task using the unified TreeOfCode approach
            result = self.process_task(user_input)
            
            # Display results based on strategy
            if strategy in ["refactor", "document", "correct"]:
                action_name = strategy.capitalize()
                if result["success"]:
                    print(Fore.GREEN + f"\n{action_name} successful!" + Style.RESET_ALL)
                    print(f"Original file: {result.get('original_file', 'Not specified')}")
                    print(f"Modified file: {result.get('solution_file', 'Not generated')}")
                    print(f"Tree file: {result.get('tree_file', 'Not generated')}")
                    print(f"Documentation: {result.get('doc_file', 'Not generated')}")
                    print(f"Visualization: {result.get('html_file', 'Not generated')}")
                else:
                    print(Fore.RED + f"\n{action_name} failed." + Style.RESET_ALL)
                    if "error" in result:
                        print(f"Error: {result['error']}")
                print(f"Results saved in: {result['work_path']}")
            
            else:  # create or generic
                solution = result["solution"]
                if solution.success:
                    print(Fore.GREEN + "\nSolution found in Tree-of-Code!" + Style.RESET_ALL)
                    print("Thought:", solution.thought)
                    print("\nGenerated code:\n", solution.code)
                    if solution.execution_result:
                        print("\nExecution results:")
                        if solution.execution_result.stdout:
                            print("Stdout:", solution.execution_result.stdout)
                        if solution.execution_result.stderr:
                            print("Stderr:", solution.execution_result.stderr)
                    print("\nReflection:", solution.reflection)
                else:
                    print(Fore.RED + "\nNo satisfactory solution found." + Style.RESET_ALL)
                    print("Best attempt:")
                    print("Thought:", solution.thought)
                    print("\nGenerated code (with issues):\n", solution.code)
                    if solution.execution_result:
                        print("\nExecution results (with issues):")
                        if solution.execution_result.stdout:
                            print("Stdout:", solution.execution_result.stdout)
                        if solution.execution_result.stderr:
                            print("Stderr:", solution.execution_result.stderr)
                    print("\nReflection:", solution.reflection)
                print(Fore.CYAN + f"\nWork files saved in: {result['work_path']}" + Style.RESET_ALL)
            
            print(Fore.YELLOW + "\nTask completed." + Style.RESET_ALL)
            
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nProcess interrupted by user." + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"\nCritical error: {e}" + Style.RESET_ALL)
            raise

    def solve_task(self, task: str, save_results: bool = True) -> TreeNode:
        self.meta.add_feedback(f"Task received: {task}", source="User", event_type="TASK")
        engine = TreeOfCodeEngine(task, self.llm_client, self.meta, self.max_tree_depth, self.max_iterations,
                                    self.thread_pool_size, checkpoint_dir=self.session_dir / "checkpoints", plugin_manager=self.plugin_manager)
        logger.info("Building Tree-of-Code...")
        try:
            engine.build_tree()
        except Exception as e:
            logger.error(f"Unexpected error during tree building: {e}")
            checkpoint_path = str(self.session_dir / f"partial_tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            engine.save_tree(checkpoint_path)
            raise e
        tree_ascii = self.visualizer.generate_ascii_tree(engine.root)
        logger.info(f"Tree structure:\n{tree_ascii}")
        solution = engine.get_final_solution()
        if solution:
            logger.info(f"Solution found at depth {solution.depth}")
            self.meta.add_feedback("Solution found successfully", source="System", event_type="SUCCESS")
            return solution
        else:
            logger.warning("No satisfactory solution found")
            self.meta.add_feedback("No satisfactory solution found", source="System", event_type="ERROR")
            leaf_nodes = engine.all_leaves
            if leaf_nodes:
                best_attempt = min(leaf_nodes, key=lambda n: (len(n.execution_result.stderr) if n.execution_result else float('inf'), n.depth))
                logger.info(f"Returning best attempt (not successful) at depth {best_attempt.depth}")
                return best_attempt
            else:
                return engine.root

def main():

    parser = argparse.ArgumentParser(description="Tree-of-Code CLI with File Processing")
    
    # Base arguments
    parser.add_argument("--task", type=str, help="Description of the task to perform", required=True)
    parser.add_argument("--file", type=str, help="Path to the file to process")
    parser.add_argument("--config", type=str, help="Path to a JSON configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Specific arguments for file operations
    group = parser.add_argument_group('File operations')
    group.add_argument("--refactor", action="store_true", help="Refactoring mode")
    group.add_argument("--document", action="store_true", help="Documentation mode")
    group.add_argument("--fix", action="store_true", help="Bug fixing mode")
    
    args = parser.parse_args()
    
    # Check if a file is needed but not provided
    # Check if a file is needed but not provided
    needs_file = args.refactor or args.document or args.fix
    if needs_file and not args.file:
        print(Fore.RED + "Error: A file must be specified with --file for refactoring, documentation, or bug fixing operations.")
        sys.exit(1)
    
    # Check if the file exists
    if args.file and not os.path.isfile(args.file):
        print(Fore.RED + f"Error: The file {args.file} does not exist.")
        sys.exit(1)
    
    try:
        print(Fore.YELLOW + "=== TREE-OF-CODE CLI ===")
        
        # Initialize the orchestrator
        orchestrator = UnifiedAgentOrchestrator()
        
        # Build the task description
        task_description = args.task
        
        # Add a prefix based on the mode
        if args.refactor:
            task_description = f"Refactor: {task_description}"
        elif args.document:
            task_description = f"Document: {task_description}"
        elif args.fix:
            task_description = f"Fix bugs in: {task_description}"
        
        # Add the file path if provided
        if args.file:
            task_description = f"{task_description} File: {args.file}"
        
        print(Fore.CYAN + f"Task: {task_description}")
        
        # Execute the processing
        result = orchestrator.process_task(task_description)
        
        # Display results based on the detected strategy
        strategy = result.get("strategy", "generic")
        print(Fore.CYAN + f"Strategy used: {strategy}")
        
        if strategy in ["refactor", "document", "correct"]:
            # Results for file-based operations
            if result.get("success", False):
                print(Fore.GREEN + "Operation successful!")
                
                if "original_file" in result:
                    print(f"Original file: {result['original_file']}")
                
                if "solution_file" in result:
                    print(f"Output file: {result['solution_file']}")
                
                if "tree_file" in result:
                    print(f"Tree file: {result['tree_file']}")
                
                if "doc_file" in result:
                    print(f"Documentation: {result['doc_file']}")
                
                if "html_file" in result:
                    print(f"Visualization: {result['html_file']}")
                
                print(f"Working directory: {result.get('work_path', '')}")
            else:
                print(Fore.RED + "Operation failed!")
                if "error" in result:
                    print(f"Error: {result['error']}")
        else:
            # Results for code creation
            solution = result.get("solution")
            if solution and solution.success:
                print(Fore.GREEN + "Solution found!")
                print("Thought:", solution.thought)
                print("\nGenerated code:")
                print(solution.code)
                
                if solution.execution_result:
                    print("\nExecution results:")
                    if solution.execution_result.stdout:
                        print("Stdout:", solution.execution_result.stdout)
                    if solution.execution_result.stderr:
                        print("Stderr:", solution.execution_result.stderr)
                
                print("\nReflection:", solution.reflection)
                print(f"\nFiles saved in: {result.get('work_path', '')}")
            else:
                print(Fore.RED + "No satisfactory solution found.")
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
