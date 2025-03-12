#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plugin DuckDuckGo pour Tree-of-Code Framework
======================================================
Ce plugin utilise le moteur de recherche DuckDuckGo pour effectuer des recherches web
et fournir des informations contextuelles au LLM pendant le processus de génération de code.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from duckduckgo_search import DDGS
import json
import os
from datetime import datetime
from abc import ABC, abstractmethod

# Configuration du logging
logger = logging.getLogger(__name__)

# Définition de la classe Plugin de base (compatible avec le framework)
class Plugin(ABC):
    """Base plugin class that matches the one in the Tree-of-Code framework."""
    name: str = "base_plugin"
    version: str = "0.1.0"
    description: str = "Base plugin class"

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

class DuckDuckGoPlugin(Plugin):
    """
    Plugin amélioré pour effectuer des recherches web avec DuckDuckGo.
    Inclut des fonctionnalités de mise en cache, d'historique et d'analyse des résultats.
    """
    name = "duckduckgo_search"
    version = "1.1.0"
    description = "Performs web searches using DuckDuckGo's search engine with caching and result analysis."

    def __init__(self):
        """Initialisation du plugin avec un cache de résultats."""
        self.search_cache = {}  # Cache pour les requêtes
        self.llm_client = None  # Client LLM pour l'analyse des résultats
        self.search_history = []  # Historique des recherches
        self.cache_dir = None  # Répertoire pour le cache persistant
        self.session_dir = None  # Répertoire de session pour les logs

    def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialise le plugin avec le contexte fourni par le framework.
        
        Args:
            context: Dictionnaire contenant les ressources du framework:
                     - plugin_manager: Le gestionnaire de plugins
                     - llm_client: Client LLM pour les requêtes au modèle
                     - meta_learner: Le système de méta-apprentissage
                     - config: La configuration du framework
                     - session_dir: Le répertoire de la session courante
        """
        logger.info("Initializing DuckDuckGoPlugin")
        
        # Récupérer les ressources du context
        if 'llm_client' in context:
            self.llm_client = context['llm_client']
            logger.info("LLM client attached to plugin")
        
        if 'plugin_manager' in context and hasattr(context['plugin_manager'], 'register_hook'):
            # Enregistrer les hooks pour les différentes phases
            pm = context['plugin_manager']
            pm.register_hook('search_query', self.search)
            pm.register_hook('analyze_search_results', self.analyze_results)
            logger.info("Registered plugin hooks: search_query, analyze_search_results")
        
        if 'session_dir' in context:
            self.session_dir = context['session_dir']
            # Créer un sous-répertoire pour les caches de recherche
            self.cache_dir = os.path.join(self.session_dir, "search_cache")
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Search cache directory: {self.cache_dir}")
        
        # Charger le cache persistant s'il existe
        self._load_cache()
        
        logger.info(f"DuckDuckGoPlugin v{self.version} initialized successfully")

    def shutdown(self) -> None:
        """Sauvegarde l'état et nettoie les ressources lors de l'arrêt."""
        logger.info("Shutting down DuckDuckGoPlugin")
        self._save_cache()
        self.search_cache = {}
        self.search_history = []
        logger.info("DuckDuckGoPlugin shutdown complete")

    def search(self, query: str, max_results: int = 10, use_cache: bool = True) -> List[Dict[str, str]]:
        """
        Effectue une recherche web en utilisant DuckDuckGo.
        
        Args:
            query: La requête de recherche
            max_results: Nombre maximum de résultats à retourner
            use_cache: Utiliser le cache pour les requêtes déjà effectuées
            
        Returns:
            Liste de dictionnaires contenant les résultats de recherche
        """
        logger.info(f"Searching for: '{query}' (max_results={max_results}, use_cache={use_cache})")
        
        # Vérifier le cache
        cache_key = f"{query}_{max_results}"
        if use_cache and cache_key in self.search_cache:
            logger.info(f"Cache hit for query: '{query}'")
            cached_results = self.search_cache[cache_key]
            
            # Ajouter à l'historique
            self.search_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "max_results": max_results,
                "num_results": len(cached_results),
                "from_cache": True
            })
            
            return cached_results
        
        try:
            # Effectuer la recherche
            results = []
            with DDGS() as ddgs:
                for i, result in enumerate(ddgs.search(query, max_results=max_results)):
                    if i >= max_results:
                        break
                    results.append(result)
            
            # Mettre en cache les résultats
            self.search_cache[cache_key] = results
            
            # Ajouter à l'historique
            self.search_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "max_results": max_results,
                "num_results": len(results),
                "from_cache": False
            })
            
            # Sauvegarder le cache périodiquement
            if len(self.search_history) % 5 == 0:  # Toutes les 5 recherches
                self._save_cache()
            
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Error performing DuckDuckGo search: {e}")
            
            # Ajouter l'échec à l'historique
            self.search_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "max_results": max_results,
                "error": str(e),
                "from_cache": False
            })
            
            return []
    
    def analyze_results(self, query: str, results: List[Dict[str, str]]) -> str:
        """
        Analyse les résultats de recherche à l'aide du LLM pour extraire les informations pertinentes.
        
        Args:
            query: La requête de recherche originale
            results: Liste des résultats de recherche à analyser
            
        Returns:
            Résumé des informations pertinentes extraites des résultats
        """
        if not results:
            return "No search results available to analyze."
        
        if not self.llm_client:
            logger.warning("LLM client not available for result analysis")
            return "Cannot analyze results: LLM client not available."
        
        # Formater les résultats pour le prompt
        formatted_results = "\n\n".join([
            f"Title: {result.get('title', 'No title')}\n"
            f"URL: {result.get('href', 'No URL')}\n"
            f"Description: {result.get('body', 'No description')}"
            for result in results[:5]  # Limiter à 5 résultats pour éviter les prompts trop longs
        ])
        
        # Créer le prompt d'analyse
        analysis_prompt = (
            f"Based on the following search results for the query '{query}', "
            f"please extract and summarize the most relevant information:\n\n"
            f"{formatted_results}\n\n"
            f"Provide a concise summary of the key information relevant to the query."
        )
        
        try:
            # Obtenir l'analyse du LLM
            analysis = self.llm_client.request(analysis_prompt)
            logger.info(f"Analysis generated for query: '{query}'")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing search results: {e}")
            return f"Error analyzing results: {str(e)}"
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique des recherches effectuées."""
        return self.search_history
    
    def clear_cache(self) -> None:
        """Vide le cache de recherche."""
        self.search_cache = {}
        logger.info("Search cache cleared")
    
    def _save_cache(self) -> None:
        """Sauvegarde le cache sur disque."""
        if not self.cache_dir:
            logger.warning("Cannot save cache: cache directory not set")
            return
        
        try:
            # Sauvegarder le cache
            cache_file = os.path.join(self.cache_dir, "search_cache.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                # Convertir en structure sérialisable (les objets peuvent contenir des types non sérialisables)
                serializable_cache = {}
                for key, value in self.search_cache.items():
                    serializable_cache[key] = value
                json.dump(serializable_cache, f, indent=2)
            
            # Sauvegarder l'historique
            history_file = os.path.join(self.cache_dir, "search_history.json")
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_history, f, indent=2)
                
            logger.info(f"Cache saved to {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error saving search cache: {e}")
    
    def _load_cache(self) -> None:
        """Charge le cache depuis le disque."""
        if not self.cache_dir:
            logger.warning("Cannot load cache: cache directory not set")
            return
        
        try:
            # Charger le cache
            cache_file = os.path.join(self.cache_dir, "search_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
                logger.info(f"Loaded {len(self.search_cache)} cached search queries")
            
            # Charger l'historique
            history_file = os.path.join(self.cache_dir, "search_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.search_history = json.load(f)
                logger.info(f"Loaded {len(self.search_history)} search history entries")
        except Exception as e:
            logger.error(f"Error loading search cache: {e}")
            # Initialiser un nouveau cache en cas d'erreur
            self.search_cache = {}
            self.search_history = []

# Variables d'exportation pour le framework
PLUGIN_CLASS = DuckDuckGoPlugin
plugin_instance = DuckDuckGoPlugin()

# Diagnostic logs
logger.info(f"DuckDuckGoPlugin module loaded successfully")
