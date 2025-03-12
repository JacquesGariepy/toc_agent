import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import File, UploadFile, Form
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil

# Import de l'orchestrateur de l'agent
from agent import UnifiedAgentOrchestrator

app = FastAPI(title="Tree-of-Code API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou listez les origines autorisées, ex: ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    task: str
    async_mode: Optional[bool] = False
    file_path: Optional[str] = None  # Path option for refactoring

class TaskResponse(BaseModel):
    success: bool
    thought: Optional[str] = None
    code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    reflection: Optional[str] = None
    error: Optional[str] = None
    strategy: Optional[str] = None  # Strategy type used
    refactored_file: Optional[str] = None
    report_file: Optional[str] = None
    work_path: Optional[str] = None

# Stockage en mémoire des résultats
task_results: Dict[str, TaskResponse] = {}

@app.post("/solve", response_model=TaskResponse)
async def solve_task_api(task_request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    async def task_runner(task_id: str, task: str, file_path: Optional[str] = None):
        try:
            orchestrator = UnifiedAgentOrchestrator()
            
            # If file_path is provided, enhance the task description
            if file_path:
                task = f"{task} File: {file_path}"
                
            # Process the task
            result = await asyncio.to_thread(orchestrator.process_task, task)
            
            # Process results based on strategy
            if "solution" in result:
                # For code generation strategies
                solution = result["solution"]
                task_results[task_id] = TaskResponse(
                    success=solution.success,
                    thought=solution.thought,
                    code=solution.code,
                    execution_result=solution.execution_result.to_dict() if solution.execution_result else None,
                    reflection=solution.reflection,
                    work_path=result.get("work_path", ""),
                    strategy=result.get("strategy", "generic")
                )
                
                # Notification websocket
                await notify_web({
                    "event": "task_completed",
                    "task_id": task_id,
                    "success": solution.success,
                    "node_id": solution.node_id if hasattr(solution, "node_id") else "",
                    "depth": solution.depth if hasattr(solution, "depth") else 0,
                    "strategy": result.get("strategy", "generic"),
                    "thought": solution.thought,
                    "code": solution.code,
                    "reflection": solution.reflection,
                    "work_path": result.get("work_path", "")
                })
            else:
                # For refactoring/documentation strategies
                task_results[task_id] = TaskResponse(
                    success=result.get("success", False),
                    work_path=result.get("work_path", ""),
                    refactored_file=result.get("refactored_file", ""),
                    report_file=result.get("report_file", ""),
                    strategy=result.get("strategy", "unknown"),
                    error=result.get("error", "")
                )
                
                # Notification websocket
                await notify_web({
                    "event": "refactoring_completed",
                    "task_id": task_id,
                    "success": result.get("success", False),
                    "strategy": result.get("strategy", "unknown"),
                    "work_path": result.get("work_path", ""),
                    "refactored_file": result.get("refactored_file", ""),
                    "report_file": result.get("report_file", "")
                })
                
        except Exception as e:
            task_results[task_id] = TaskResponse(success=False, error=str(e))
            # Notification of error
            await notify_web({
                "event": "error",
                "task_id": task_id,
                "error": str(e)
            })
    
    background_tasks.add_task(task_runner, task_id, task_request.task, task_request.file_path)
    return TaskResponse(success=True, thought="Task submitted, check results later.")

# Correction de l'erreur de syntaxe dans la fonction upload_and_process
@app.post("/upload-and-process", response_model=TaskResponse)
async def upload_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task: str = Form(...)
):
    task_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    # Create temp directory for the uploaded file
    temp_dir = tempfile.mkdtemp(prefix="toc_upload_")
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        return TaskResponse(success=False, error=f"Error saving uploaded file: {str(e)}")
    
    # Modifiez la fonction process_uploaded_file pour gérer le cas d'une tâche "create"/"solve"
    async def process_uploaded_file(task_id: str, task: str, file_path: str, temp_dir: str):
        try:
            orchestrator = UnifiedAgentOrchestrator()
            
            # Enhance task with file path information
            enhanced_task = f"{task} File: {file_path}"
            
            # Process the task
            result = await asyncio.to_thread(orchestrator.process_task, enhanced_task)
            
            # Detect task type from strategy
            strategy = result.get("strategy", "unknown")
            
            # Store and notify results based on strategy
            if strategy in ["refactor", "document", "correct"]:
                # File-based operations with expected outputs
                task_results[task_id] = TaskResponse(
                    success=result.get("success", False),
                    work_path=result.get("work_path", ""),
                    refactored_file=result.get("refactored_file", ""),
                    report_file=result.get("report_file", ""),
                    strategy=strategy,
                    error=result.get("error", "")
                )
                
                await notify_web({
                    "event": "refactoring_completed",
                    "task_id": task_id,
                    "success": result.get("success", False),
                    "strategy": strategy,
                    "work_path": result.get("work_path", ""),
                    "refactored_file": result.get("refactored_file", ""),
                    "report_file": result.get("report_file", "")
                })
            elif "solution" in result:
                # Code generation operations
                solution = result["solution"]
                task_results[task_id] = TaskResponse(
                    success=solution.success,
                    thought=solution.thought,
                    code=solution.code,
                    execution_result=solution.execution_result.to_dict() if solution.execution_result else None,
                    reflection=solution.reflection,
                    work_path=result.get("work_path", ""),
                    strategy=strategy
                )
                
                # Notification websocket
                await notify_web({
                    "event": "task_completed",
                    "task_id": task_id,
                    "success": solution.success,
                    "node_id": solution.node_id if hasattr(solution, "node_id") else "",
                    "depth": solution.depth if hasattr(solution, "depth") else 0,
                    "strategy": strategy,
                    "thought": solution.thought,
                    "code": solution.code,
                    "reflection": solution.reflection,
                    "work_path": result.get("work_path", "")
                })
            else:
                # Unhandled strategy
                task_results[task_id] = TaskResponse(
                    success=False,
                    error=f"Unexpected result format for strategy: {strategy}",
                    strategy=strategy
                )
                
                await notify_web({
                    "event": "error",
                    "task_id": task_id,
                    "error": f"Unexpected result format for strategy: {strategy}",
                    "strategy": strategy
                })
        except Exception as e:
            task_results[task_id] = TaskResponse(success=False, error=str(e))
            await notify_web({
                "event": "error",
                "task_id": task_id,
                "error": str(e)
            })
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary directory: {cleanup_error}")
    
    background_tasks.add_task(process_uploaded_file, task_id, task, file_path, temp_dir)
    return TaskResponse(success=True, thought="File uploaded and processing started.")

@app.get("/result/{task_id}", response_model=TaskResponse)
def get_result(task_id: str):
    result = task_results.get(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task result not found.")
    return result

# WebSocket Manager for real-time notifications
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message to WebSocket: {e}")

manager = ConnectionManager()

async def notify_web(message: dict):
    try:
        await manager.send_message(message)
    except Exception as e:
        print(f"Error in notify_web: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
