export default {
    root: './src',
    server: {
      watch: {
        ignored: [
            /logs\/.*/,             // ignore tous les fichiers dans le dossier logs et ses sous-dossiers
            /\.pytest_cache\/.*/,    // ignore le dossier .pytest_cache
            /\.toc_cache\/.*/,       // ignore le dossier .toc_cache
            /memory\.sqlite.*/,       // ignore memory.sqlite et tout ce qui commence par memory.sqlite
            /.*_autosave\.json$/     // ignore tous les fichiers se terminant par _autosave.json
          ]
      }
    }
  }
  
