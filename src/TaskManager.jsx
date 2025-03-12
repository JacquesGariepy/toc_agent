import React, { useState, useEffect } from 'react';
import useWebSocket from 'react-use-websocket';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion, AnimatePresence } from 'framer-motion';

function TaskManager() {
  const [task, setTask] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitResponse, setSubmitResponse] = useState('');
  const [logs, setLogs] = useState([]);
  const [file, setFile] = useState(null);
  const [uploadMode, setUploadMode] = useState(false);
  const [filePath, setFilePath] = useState('');
  const [taskType, setTaskType] = useState('solve'); // 'solve', 'refactor', 'document', 'correct'

  // Connexion WebSocket
  const { lastJsonMessage } = useWebSocket('ws://localhost:8000/ws', {
    onOpen: () => console.log("WebSocket connecté"),
    shouldReconnect: () => true,
  });

  // Ajout des messages reçus aux logs
  useEffect(() => {
    if (lastJsonMessage) {
      console.log("Message reçu :", lastJsonMessage);
      setLogs(prevLogs => [lastJsonMessage, ...prevLogs]);
      if (lastJsonMessage.event === 'task_completed' || 
          lastJsonMessage.event === 'refactoring_completed' || 
          lastJsonMessage.event === 'error') {
        setLoading(false);
      }
    }
  }, [lastJsonMessage]);

  // Gestion des changements de fichier
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  // Bascule entre mode upload et chemin de fichier
  const handleModeChange = (e) => {
    setUploadMode(e.target.value === 'upload');
    if (e.target.value === 'upload') {
      setFilePath('');
    } else {
      setFile(null);
    }
  };

  // Gestion du type de tâche
  const handleTaskTypeChange = (e) => {
    const newTaskType = e.target.value;
    setTaskType(newTaskType);
    if (newTaskType === 'solve') {
      setFile(null);
      setFilePath('');
      setUploadMode(false);
    }
  };

  // Soumission du formulaire
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSubmitResponse('');
    
    try {
      let response;
      if (taskType === 'solve') {
        const payload = { task };
        response = await fetch('http://localhost:8000/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else if (uploadMode && file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('task', `${taskType}: ${task}`);
        response = await fetch('http://localhost:8000/upload-and-process', {
          method: 'POST',
          body: formData,
        });
      } else {
        const payload = { task: `${taskType}: ${task}`, file_path: filePath || undefined };
        response = await fetch('http://localhost:8000/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }
      const data = await response.json();
      setSubmitResponse(data.thought || 'Tâche soumise avec succès');
    } catch (error) {
      console.error("Erreur lors de la soumission :", error);
      setSubmitResponse('Erreur : ' + error.message);
      setLoading(false);
    }
  };

  const needsFileInput = taskType === 'refactor' || taskType === 'document' || taskType === 'correct';

  return (
    <div className="p-4 font-sans max-w-5xl mx-auto bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-8 text-gray-800">Tree-of-Code Manager</h1>

      {/* Formulaire */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white p-6 rounded-lg shadow-lg mb-8"
      >
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label className="block text-gray-700 font-semibold mb-2">Type de tâche :</label>
            <select
              value={taskType}
              onChange={handleTaskTypeChange}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="solve">Créer un nouveau code</option>
              <option value="refactor">Refactoriser un code existant</option>
              <option value="document">Documenter un code existant</option>
              <option value="correct">Corriger des erreurs dans un code</option>
            </select>
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 font-semibold mb-2">Description de la tâche :</label>
            <textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows="4"
              placeholder="Décrivez votre tâche..."
              required
            />
          </div>

          {needsFileInput && (
            <div className="mb-6">
              <label className="block text-gray-700 font-semibold mb-2">Source du code :</label>
              <div className="flex space-x-4 mb-4">
                <label className="inline-flex items-center cursor-pointer">
                  <input
                    type="radio"
                    value="path"
                    checked={!uploadMode}
                    onChange={handleModeChange}
                    className="form-radio text-blue-500"
                  />
                  <span className="ml-2">Fournir un chemin de fichier</span>
                </label>
                <label className="inline-flex items-center cursor-pointer">
                  <input
                    type="radio"
                    value="upload"
                    checked={uploadMode}
                    onChange={handleModeChange}
                    className="form-radio text-blue-500"
                  />
                  <span className="ml-2">Uploader un fichier</span>
                </label>
              </div>

              {uploadMode ? (
                <div>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    className="w-full p-2 border border-gray-300 rounded-md file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                    accept=".py,.js,.ts,.java,.c,.cpp,.html,.css"
                  />
                  {file && <p className="mt-2 text-sm text-gray-500">Fichier sélectionné : {file.name}</p>}
                </div>
              ) : (
                <input
                  type="text"
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="/chemin/vers/votre/fichier.py"
                />
              )}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || (needsFileInput && !uploadMode && !filePath) || (needsFileInput && uploadMode && !file)}
            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-4 rounded-md transition duration-300 ease-in-out disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Traitement en cours...
              </>
            ) : (
              'Soumettre la tâche'
            )}
          </button>
        </form>
      </motion.div>

      {/* Réponse de soumission */}
      {submitResponse && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8 p-4 bg-green-100 border border-green-200 rounded-md text-green-800"
        >
          {submitResponse}
        </motion.div>
      )}

      {/* Logs en temps réel */}
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Logs en temps réel</h2>
      <div className="space-y-4">
        <AnimatePresence>
          {logs.map((log, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="border rounded-lg shadow-sm p-4 bg-white"
            >
              <div className="flex justify-between items-center mb-2">
                <span className={`font-semibold ${
                  log.event === 'task_completed' || log.event === 'refactoring_completed' ? 'text-green-600' :
                  log.event === 'error' ? 'text-red-600' : 'text-blue-600'
                }`}>
                  {log.event === 'task_completed' ? '✅ Tâche terminée' :
                   log.event === 'refactoring_completed' ? '🔄 Refactorisation terminée' :
                   log.event === 'error' ? '❌ Erreur' : log.event}
                </span>
                {log.strategy && (
                  <span className="px-2 py-1 rounded-full text-xs bg-blue-100 text-blue-800">
                    {log.strategy}
                  </span>
                )}
              </div>

              {log.task_id && <p className="text-sm mb-1"><strong>ID de la tâche :</strong> {log.task_id}</p>}
              {log.success !== undefined && (
                <p className="text-sm mb-1">
                  <strong>Statut :</strong>
                  <span className={log.success ? "text-green-600" : "text-red-600"}>
                    {log.success ? " Succès" : " Échec"}
                  </span>
                </p>
              )}

              {log.refactored_file && <p className="text-sm mb-1"><strong>Fichier refactorisé :</strong> {log.refactored_file}</p>}
              {log.report_file && <p className="text-sm mb-1"><strong>Rapport :</strong> {log.report_file}</p>}
              {log.work_path && <p className="text-sm mb-1"><strong>Chemin de travail :</strong> {log.work_path}</p>}

              {log.error && (
                <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-red-700">
                  <strong>Erreur :</strong> {log.error}
                </div>
              )}

              {log.code && (
                <details className="mt-3">
                  <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                    Voir le code généré
                  </summary>
                  <div className="mt-2">
                    <SyntaxHighlighter language="python" style={tomorrow}>
                      {log.code}
                    </SyntaxHighlighter>
                  </div>
                </details>
              )}

              {log.thought && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                    Voir le raisonnement
                  </summary>
                  <div className="mt-2 p-3 bg-gray-50 rounded">
                    {log.thought}
                  </div>
                </details>
              )}

              {log.reflection && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                    Voir la réflexion
                  </summary>
                  <div className="mt-2 p-3 bg-gray-50 rounded">
                    {log.reflection}
                  </div>
                </details>
              )}

              <details className="mt-2">
                <summary className="cursor-pointer text-gray-500 text-sm">
                  Voir tous les détails
                </summary>
                <pre className="mt-2 p-2 bg-gray-50 rounded text-xs overflow-auto">
                  {JSON.stringify(log, null, 2)}
                </pre>
              </details>
            </motion.div>
          ))}
        </AnimatePresence>

        {logs.length === 0 && (
          <div className="text-center p-6 text-gray-500 bg-white rounded-lg shadow-sm">
            Aucun log pour le moment. Soumettez une tâche pour voir les logs ici.
          </div>
        )}
      </div>
    </div>
  );
}

export default TaskManager;
