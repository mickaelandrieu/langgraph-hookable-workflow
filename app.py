# Installation requise:
# pip install langgraph langchain-core scikit-learn numpy sentence-transformers

from typing import TypedDict, List, Dict, Any, Optional, Callable, Annotated
from abc import ABC, abstractmethod
import asyncio
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph imports
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig
from operator import add

# Fonction de merge pour les dictionnaires
def merge_dicts(a: dict, b: dict) -> dict:
    """Merge deux dictionnaires de manière sûre"""
    result = a.copy() if a else {}
    if b:
        result.update(b)
    return result


# ============================================================================
# ÉTAT DU GRAPHE AVEC ANNOTATIONS
# ============================================================================

class SearchState(TypedDict):
    """État partagé du graphe LangGraph avec annotations pour la concurrence"""
    # Données de requête (pas de conflit)
    original_query: str
    processed_query: str
    query_expansion: List[str]
    query_intent: str
    
    # Résultats de recherche - ANNOTATED pour éviter les conflits concurrents
    vector_results: Annotated[List[Dict[str, Any]], add]
    keyword_results: Annotated[List[Dict[str, Any]], add]
    combined_results: List[Dict[str, Any]]
    final_results: List[Dict[str, Any]]
    
    # Métadonnées - ANNOTATED avec fonction de merge personnalisée
    processing_metadata: Annotated[Dict[str, Any], merge_dicts]
    performance_metrics: Annotated[Dict[str, float], merge_dicts]
    
    # IMPORTANT: Réponse LLM finale
    llm_response: str


# ============================================================================
# INTERFACES POUR LES HOOKS
# ============================================================================

class PreprocessorHook(ABC):
    """Interface pour les hooks de préprocessing"""
    
    @abstractmethod
    async def clean_query(self, query: str) -> str:
        """Nettoyage de la requête"""
        pass
    
    @abstractmethod
    async def expand_query(self, query: str) -> List[str]:
        """Expansion de la requête"""
        pass
    
    @abstractmethod
    async def detect_intent(self, query: str) -> str:
        """Détection d'intention"""
        pass


class SearcherHook(ABC):
    """Interface pour les hooks de recherche"""
    
    @abstractmethod
    async def search(self, queries: List[str], method: str) -> List[Dict[str, Any]]:
        """Recherche avec méthode spécifiée"""
        pass


class PostprocessorHook(ABC):
    """Interface pour les hooks de post-processing"""
    
    @abstractmethod
    async def combine_results(self, vector_results: List[Dict], 
                            keyword_results: List[Dict]) -> List[Dict]:
        """Combinaison des résultats"""
        pass
    
    @abstractmethod
    async def rerank_results(self, results: List[Dict], 
                           query: str, intent: str) -> List[Dict]:
        """Re-ranking des résultats"""
        pass


class LLMHook(ABC):
    """Interface pour les hooks LLM"""
    
    @abstractmethod
    async def generate_response(self, query: str, context: str, intent: str) -> str:
        """Génération de réponse"""
        pass


# ============================================================================
# IMPLÉMENTATIONS PAR DÉFAUT
# ============================================================================

class DefaultPreprocessor(PreprocessorHook):
    """Implémentation par défaut du préprocessing"""
    
    async def clean_query(self, query: str) -> str:
        """Nettoyage basique"""
        import re
        cleaned = re.sub(r'[^\w\s\-\.éèàùç]', '', query, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned
    
    async def expand_query(self, query: str) -> List[str]:
        """Expansion basique"""
        expansions = [query]
        if "ml" in query:
            expansions.append(query.replace("ml", "machine learning"))
        if "python" in query:
            expansions.append(query + " programmation")
        return expansions
    
    async def detect_intent(self, query: str) -> str:
        """Détection d'intention basique"""
        if any(word in query for word in ["qu'est", "définir"]):
            return "definition"
        return "general"


class DefaultSearcher(SearcherHook):
    """Implémentation par défaut de la recherche"""
    
    def __init__(self):
        # Chargement des modèles d'embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-transformers chargé")
        except ImportError:
            self.embedding_model = self._mock_embedding_model()
            print("⚠️ Utilisation d'embeddings simulés")
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Documents de test
        self.documents = [
            {"id": 1, "content": "Python est un langage de programmation polyvalent utilisé en data science", 
             "title": "Introduction Python", "category": "programming"},
            {"id": 2, "content": "Machine learning permet aux ordinateurs d'apprendre sans programmation explicite", 
             "title": "ML Basics", "category": "ai"},
            {"id": 3, "content": "Les réseaux de neurones artificiels s'inspirent du cerveau humain", 
             "title": "Neural Networks", "category": "ai"},
            {"id": 4, "content": "LangGraph facilite la création de workflows complexes avec des LLMs", 
             "title": "LangGraph Guide", "category": "tools"},
            {"id": 5, "content": "La recherche vectorielle utilise des embeddings pour la similarité", 
             "title": "Vector Search", "category": "search"}
        ]
        
        # Préparation des données
        self.contents = [doc['content'] for doc in self.documents]
        self.doc_embeddings = self.embedding_model.encode(self.contents)
        self.tfidf_matrix = self.tfidf.fit_transform(self.contents)
    
    def _mock_embedding_model(self):
        """Modèle d'embedding simulé"""
        class MockModel:
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = []
                for text in texts:
                    vector = np.array([hash(text + str(i)) % 1000 / 1000.0 for i in range(384)])
                    embeddings.append(vector)
                return np.array(embeddings)
        return MockModel()
    
    async def search(self, queries: List[str], method: str) -> List[Dict[str, Any]]:
        """Recherche selon la méthode spécifiée"""
        if method == "vector":
            return await self._vector_search(queries)
        elif method == "keyword":
            return await self._keyword_search(queries)
        else:
            raise ValueError("Méthode de recherche inconnue: " + method)
    
    async def _vector_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Recherche vectorielle"""
        all_results = {}
        for i, query in enumerate(queries):
            query_embedding = self.embedding_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
            
            weight = 1.0 / (i + 1)
            for idx, similarity in enumerate(similarities):
                doc_id = self.documents[idx]["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "document": self.documents[idx],
                        "score": 0,
                        "method": "vector"
                    }
                all_results[doc_id]["score"] += similarity * weight
        
        results = list(all_results.values())
        if results:
            max_score = max(r["score"] for r in results)
            for r in results:
                r["score"] = r["score"] / max_score if max_score > 0 else 0
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]
    
    async def _keyword_search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Recherche par mots-clés"""
        all_results = {}
        for i, query in enumerate(queries):
            try:
                query_tfidf = self.tfidf.transform([query])
                similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
                
                weight = 1.0 / (i + 1)
                for idx, similarity in enumerate(similarities):
                    if similarity > 0:
                        doc_id = self.documents[idx]["id"]
                        if doc_id not in all_results:
                            all_results[doc_id] = {
                                "document": self.documents[idx],
                                "score": 0,
                                "method": "keyword"
                            }
                        all_results[doc_id]["score"] += similarity * weight
            except Exception as e:
                print("   ⚠️ Erreur TF-IDF pour '" + query + "': " + str(e))
                continue
        
        results = list(all_results.values())
        if results:
            max_score = max(r["score"] for r in results)
            for r in results:
                r["score"] = r["score"] / max_score if max_score > 0 else 0
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]


class DefaultPostprocessor(PostprocessorHook):
    """Implémentation par défaut du post-processing"""
    
    async def combine_results(self, vector_results: List[Dict], 
                            keyword_results: List[Dict]) -> List[Dict]:
        """Combinaison standard"""
        alpha = 0.6
        combined_scores = {}
        
        for result in vector_results:
            doc_id = result["document"]["id"]
            combined_scores[doc_id] = {
                "document": result["document"],
                "vector_score": result["score"],
                "keyword_score": 0.0,
                "combined_score": alpha * result["score"]
            }
        
        for result in keyword_results:
            doc_id = result["document"]["id"]
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = result["score"]
                combined_scores[doc_id]["combined_score"] += (1 - alpha) * result["score"]
            else:
                combined_scores[doc_id] = {
                    "document": result["document"],
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "combined_score": (1 - alpha) * result["score"]
                }
        
        return list(combined_scores.values())
    
    async def rerank_results(self, results: List[Dict], 
                           query: str, intent: str) -> List[Dict]:
        """Re-ranking basique"""
        for result in results:
            boost = 1.0
            if intent == "definition":
                if any(word in result["document"]["content"].lower() 
                       for word in ["définition", "est", "permet"]):
                    boost = 1.2
            
            result["combined_score"] *= boost
            result["intent_boost"] = boost
        
        return sorted(results, key=lambda x: x["combined_score"], reverse=True)


class DefaultLLM(LLMHook):
    """Implémentation par défaut du LLM"""
    
    async def generate_response(self, query: str, context: str, intent: str) -> str:
        """Génération avec OpenAI ou fallback"""
        try:
            # Tentative d'utilisation d'OpenAI
            try:
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage, SystemMessage
                
                llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    max_tokens=500
                )
                
                system_prompt = """Tu es un assistant expert en recherche d'information. 
Tu dois fournir une réponse précise et concise basée uniquement sur les documents fournis.
Structure ta réponse de manière claire et cite les sources pertinentes."""
                
                user_prompt = "Question de l'utilisateur: " + query + "\n"
                user_prompt += "Intention détectée: " + intent + "\n\n"
                user_prompt += "Documents trouvés par la recherche hybride:\n" + context + "\n\n"
                user_prompt += "Fournis une réponse complète et structurée en te basant sur ces documents."
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                
                response = llm.invoke(messages)
                return response.content
                
            except ImportError:
                # Fallback si OpenAI non disponible
                return self._fallback_response(query, context, intent)
                
        except Exception as e:
            print("   ❌ Erreur LLM: " + str(e))
            return self._error_response(query, context, str(e))
    
    def _fallback_response(self, query: str, context: str, intent: str) -> str:
        """Réponse de fallback"""
        response_title = "**Réponse simulée** (OpenAI non configuré)\n\n"
        question_part = "Pour votre question: \"" + query + "\"\n\n"
        context_part = "Basé sur la recherche hybride, voici les documents les plus pertinents:\n\n" + context + "\n\n"
        summary_part = "**Résumé**: D'après ces sources, " + query + " fait référence à des concepts importants en technologie et intelligence artificielle.\n\n"
        note_part = "*Note: Pour obtenir une réponse générée par IA, configurez votre clé API OpenAI*"
        
        return response_title + question_part + context_part + summary_part + note_part
    
    def _error_response(self, query: str, context: str, error: str) -> str:
        """Réponse en cas d'erreur"""
        error_intro = "**Erreur lors de l'appel LLM**: " + error + "\n\n"
        fallback_intro = "**Réponse de secours basée sur la recherche:**\n\n"
        config_note = "\n\nVeuillez vérifier votre configuration OpenAI."
        
        return error_intro + fallback_intro + context + config_note


# ============================================================================
# GESTIONNAIRE DE HOOKS
# ============================================================================

class HookManager:
    """Gestionnaire centralisé des hooks"""
    
    def __init__(self):
        # Hooks par défaut
        self.preprocessor = DefaultPreprocessor()
        self.searcher = DefaultSearcher()
        self.postprocessor = DefaultPostprocessor()
        self.llm = DefaultLLM()
        
        print("✅ Gestionnaire de hooks initialisé avec implémentations par défaut")
    
    def register_preprocessor(self, hook: PreprocessorHook, name: str = "custom"):
        """Enregistrer un hook de préprocessing"""
        self.preprocessor = hook
        print("🔧 Hook preprocessor '" + name + "' enregistré")
    
    def register_searcher(self, hook: SearcherHook, name: str = "custom"):
        """Enregistrer un hook de recherche"""
        self.searcher = hook
        print("🔧 Hook searcher '" + name + "' enregistré")
    
    def register_postprocessor(self, hook: PostprocessorHook, name: str = "custom"):
        """Enregistrer un hook de post-processing"""
        self.postprocessor = hook
        print("🔧 Hook postprocessor '" + name + "' enregistré")
    
    def register_llm(self, hook: LLMHook, name: str = "custom"):
        """Enregistrer un hook LLM"""
        self.llm = hook
        print("🔧 Hook LLM '" + name + "' enregistré")


# ============================================================================
# NŒUDS DU GRAPHE UTILISANT LES HOOKS
# ============================================================================

def preprocess_node(state: SearchState, config: RunnableConfig) -> SearchState:
    """NŒUD 1: Pré-traitement avec hooks"""
    print("🔧 [NODE] Pré-traitement: '" + state['original_query'] + "'")
    
    start_time = time.time()
    hook_manager = config.get("configurable", {}).get("hook_manager")
    
    async def process():
        # Utilisation des hooks
        cleaned = await hook_manager.preprocessor.clean_query(state["original_query"])
        expansions = await hook_manager.preprocessor.expand_query(cleaned)
        intent = await hook_manager.preprocessor.detect_intent(state["original_query"])
        
        print("   ✓ Expansions: " + str(expansions))
        print("   ✓ Intention: " + intent)
        
        return {
            "processed_query": cleaned,
            "query_expansion": expansions,
            "query_intent": intent,
            "processing_metadata": {"preprocess_time": time.time() - start_time}
        }
    
    return asyncio.run(process())


def vector_search_node(state: SearchState, config: RunnableConfig) -> SearchState:
    """NŒUD 2: Recherche vectorielle avec hooks"""
    print("🔍 [NODE] Recherche vectorielle...")
    
    start_time = time.time()
    hook_manager = config.get("configurable", {}).get("hook_manager")
    
    async def search():
        results = await hook_manager.searcher.search(state["query_expansion"], "vector")
        print("   ✓ " + str(len(results)) + " résultats vectoriels")
        
        return {
            "vector_results": results,
            "performance_metrics": {"vector_search_time": time.time() - start_time}
        }
    
    return asyncio.run(search())


def keyword_search_node(state: SearchState, config: RunnableConfig) -> SearchState:
    """NŒUD 3: Recherche par mots-clés avec hooks"""
    print("📝 [NODE] Recherche par mots-clés...")
    
    start_time = time.time()
    hook_manager = config.get("configurable", {}).get("hook_manager")
    
    async def search():
        results = await hook_manager.searcher.search(state["query_expansion"], "keyword")
        print("   ✓ " + str(len(results)) + " résultats par mots-clés")
        
        return {
            "keyword_results": results,
            "performance_metrics": {"keyword_search_time": time.time() - start_time}
        }
    
    return asyncio.run(search())


def combine_results_node(state: SearchState, config: RunnableConfig) -> SearchState:
    """NŒUD 4: Combinaison avec hooks"""
    print("🔄 [NODE] Combinaison des résultats...")
    
    start_time = time.time()
    hook_manager = config.get("configurable", {}).get("hook_manager")
    
    async def combine():
        # Combinaison avec hook
        combined = await hook_manager.postprocessor.combine_results(
            state["vector_results"], 
            state["keyword_results"]
        )
        
        # Re-ranking avec hook
        reranked = await hook_manager.postprocessor.rerank_results(
            combined, state["processed_query"], state["query_intent"]
        )
        
        print("   ✓ " + str(len(reranked)) + " résultats combinés")
        
        return {
            "combined_results": reranked,
            "final_results": reranked[:3],
            "performance_metrics": {"combine_time": time.time() - start_time}
        }
    
    return asyncio.run(combine())


def llm_response_node(state: SearchState, config: RunnableConfig) -> SearchState:
    """NŒUD 5: Génération LLM avec hooks"""
    print("🤖 [NODE] Génération réponse LLM...")
    
    start_time = time.time()
    hook_manager = config.get("configurable", {}).get("hook_manager")
    
    async def generate():
        # Construction du contexte
        context_docs = []
        for result in state["final_results"]:
            doc = result["document"]
            score = result.get("combined_score", 0)
            score_text = "{:.3f}".format(score)
            context_docs.append("- **" + doc['title'] + "** (score: " + score_text + "): " + doc['content'])
        
        context = "\n".join(context_docs)
        
        # Génération avec hook
        response = await hook_manager.llm.generate_response(
            state["original_query"], context, state["query_intent"]
        )
        
        print("   ✅ Réponse LLM générée")
        
        return {
            "llm_response": response,
            "performance_metrics": {"llm_time": time.time() - start_time}
        }
    
    return asyncio.run(generate())


# ============================================================================
# CRÉATION DU GRAPHE AVEC HOOKS
# ============================================================================

def create_hookable_hybrid_search_graph(hook_manager: HookManager) -> StateGraph:
    """Création du graphe avec système de hooks"""
    
    graph = StateGraph(SearchState)
    
    # Ajout des nœuds utilisant les hooks
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("keyword_search", keyword_search_node)
    graph.add_node("combine", combine_results_node)
    graph.add_node("llm_response", llm_response_node)
    
    # Définition du flux
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "vector_search")
    graph.add_edge("preprocess", "keyword_search")
    graph.add_edge(["vector_search", "keyword_search"], "combine")
    graph.add_edge("combine", "llm_response")
    graph.set_finish_point("llm_response")
    
    return graph.compile()


# ============================================================================
# FONCTION PRINCIPALE AVEC HOOKS
# ============================================================================

async def run_hookable_hybrid_search(query: str, hook_manager: HookManager = None) -> SearchState:
    """Fonction principale avec système de hooks"""
    if hook_manager is None:
        hook_manager = HookManager()
    
    print("\n🚀 DÉMARRAGE RECHERCHE HYBRIDE AVEC HOOKS")
    print("=" * 60)
    print("Requête: '" + query + "'")
    print("=" * 60)
    
    # Création du graphe avec hooks
    app = create_hookable_hybrid_search_graph(hook_manager)
    
    # État initial
    initial_state = SearchState(
        original_query=query,
        processed_query="",
        query_expansion=[],
        query_intent="",
        vector_results=[],
        keyword_results=[],
        combined_results=[],
        final_results=[],
        processing_metadata={},
        performance_metrics={},
        llm_response=""
    )
    
    # Configuration avec hook manager
    config = RunnableConfig(
        configurable={"hook_manager": hook_manager}
    )
    
    try:
        result = await app.ainvoke(initial_state, config)
        
        print("\n" + "=" * 60)
        print("📋 RÉSULTATS FINAUX:")
        print("=" * 60)
        
        print("\n🎯 Top 3 documents:")
        for i, res in enumerate(result["final_results"], 1):
            doc = res["document"]
            score_text = "{:.3f}".format(res["combined_score"])
            print("   " + str(i) + ". " + doc['title'] + " (score: " + score_text + ")")
            print("      " + doc['content'][:80] + "...")
        
        print("\n🤖 RÉPONSE LLM:")
        print(result["llm_response"])
        
        print("\n⏱️ Métriques de performance:")
        for metric, value in result["performance_metrics"].items():
            if isinstance(value, (int, float)):
                time_text = "{:.3f}s".format(value)
                print("   " + metric + ": " + time_text)
            else:
                print("   " + metric + ": " + str(value))
        
        return result
        
    except Exception as e:
        print("❌ Erreur: " + str(e))
        raise


# ============================================================================
# EXEMPLE D'UTILISATION AVEC HOOKS PERSONNALISÉS
# ============================================================================

# Hook personnalisé pour démonstration
class CustomPreprocessor(PreprocessorHook):
    """Préprocesseur personnalisé avec validation avancée"""
    
    async def clean_query(self, query: str) -> str:
        # Validation de longueur
        if len(query) < 3:
            return query  # Pas de nettoyage pour les requêtes courtes
        
        # Nettoyage avancé
        import re
        query = re.sub(r'[^\w\s\-\?]', '', query.lower().strip())
        return query
    
    async def expand_query(self, query: str) -> List[str]:
        # Expansion avec synonymes étendus
        expansions = [query]
        
        synonyms = {
            "python": ["python", "programming", "développement", "langage"],
            "ml": ["machine learning", "apprentissage automatique", "intelligence artificielle"],
            "ai": ["intelligence artificielle", "artificial intelligence", "IA"]
        }
        
        for term, syns in synonyms.items():
            if term in query.lower():
                for syn in syns:
                    exp = query.replace(term, syn)
                    if exp not in expansions:
                        expansions.append(exp)
        
        return expansions[:5]  # Limiter à 5
    
    async def detect_intent(self, query: str) -> str:
        # Détection d'intention plus sophistiquée
        patterns = {
            "definition": ["qu'est-ce", "définir", "définition", "c'est quoi"],
            "how_to": ["comment", "étapes", "procédure", "tutoriel"],
            "comparison": ["différence", "versus", "vs", "comparer"],
            "example": ["exemple", "cas d'usage", "illustration"]
        }
        
        query_lower = query.lower()
        for intent, words in patterns.items():
            if any(word in query_lower for word in words):
                return intent
        
        return "general"


async def demo_with_custom_hooks():
    """Démonstration avec hooks personnalisés"""
    
    print("🔧 DÉMONSTRATION AVEC HOOKS PERSONNALISÉS")
    print("=" * 60)
    
    # Création du gestionnaire de hooks
    hook_manager = HookManager()
    
    # Enregistrement d'un hook personnalisé
    hook_manager.register_preprocessor(CustomPreprocessor(), "advanced")
    
    # Test avec hook personnalisé
    result = await run_hookable_hybrid_search(
        "qu'est-ce que Python pour le ML ?", 
        hook_manager
    )
    
    print("\n🎯 Métadonnées de traitement:")
    for key, value in result["processing_metadata"].items():
        print("   " + key + ": " + str(value))


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

async def main():
    """Fonction principale de test"""
    
    # Test 1: Avec hooks par défaut
    print("=" * 80)
    print("TEST 1: HOOKS PAR DÉFAUT")
    print("=" * 80)
    
    result1 = await run_hookable_hybrid_search("qu'est-ce que Python ?")
    
    print("\n" + "=" * 80)
    input("Appuyez sur Entrée pour tester les hooks personnalisés...")
    
    # Test 2: Avec hooks personnalisés
    print("\n" + "=" * 80)
    print("TEST 2: HOOKS PERSONNALISÉS")
    print("=" * 80)
    
    await demo_with_custom_hooks()


if __name__ == "__main__":
    print("🔍 SYSTÈME HYBRID SEARCH AVEC ARCHITECTURE DE HOOKS")
    print("Ce système permet de remplacer chaque étape par une implémentation personnalisée")
    
    asyncio.run(main())
