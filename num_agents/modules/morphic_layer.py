"""
MorphicLayer - Morphic Reasoning and Goal-Oriented Agent System

This module implements the Morphic Universe system with:
- U₀: Base Morphic Universe (objects, states, morphisms)
- U_G: Goal Universe (objectives with boolean criteria)
- U_Φ: Morphic Flux (plan generation and execution)
- U_Ψ: Logical Selector (filtering and routing)
- U_mem: Cognitive Memory (traces and vector memory)
- RRLA: Reflection Reasoning Loop Agent
- CoR: Chain of Reasoning
- Persona: Agent personality system (Nümtema)

Postulates G1-G6:
- G1: An objective is a defined goal
- G2: An objective is a boolean function
- G3: Objectives are evaluable on objects
- G4: Objectives guide fluxes
- G5: Objectives are memorizable
- G6: Objectives are dynamically generatable

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import uuid

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class MorphicLayerException(NumAgentsException):
    """Base exception for MorphicLayer errors."""

    pass


class ObjectiveError(MorphicLayerException):
    """Exception raised when objective operations fail."""

    pass


class FluxError(MorphicLayerException):
    """Exception raised when flux operations fail."""

    pass


class ReasoningError(MorphicLayerException):
    """Exception raised when reasoning operations fail."""

    pass


class MemoryError(MorphicLayerException):
    """Exception raised when memory operations fail."""

    pass


# ============================================================================
# U₀: Base Morphic Universe - Objects, States, Morphisms
# ============================================================================


class EtatObjet(Enum):
    """State enumeration for reactive objects."""

    INACTIF = "inactif"
    ACTIF = "actif"
    EN_COURS = "en_cours"
    TERMINE = "termine"
    ERREUR = "erreur"
    SUSPENDU = "suspendu"


@dataclass
class ObjetReactif:
    """
    Reactive object in the Morphic Universe.

    Represents an object with state, history, and level tracking.
    """

    nom: str
    etat: EtatObjet = EtatObjet.INACTIF
    niveau: int = 0
    donnees: Dict[str, Any] = field(default_factory=dict)
    historique: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_creation: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def changer_etat(self, nouvel_etat: EtatObjet, raison: Optional[str] = None) -> None:
        """
        Change object state and record in history.

        Args:
            nouvel_etat: New state
            raison: Optional reason for state change
        """
        ancien_etat = self.etat
        self.etat = nouvel_etat

        # Record in history
        self.historique.append(
            {
                "timestamp": time.time(),
                "ancien_etat": ancien_etat.value,
                "nouvel_etat": nouvel_etat.value,
                "raison": raison,
                "niveau": self.niveau,
            }
        )

    def incrementer_niveau(self) -> None:
        """Increment object level."""
        self.niveau += 1

    def set_donnee(self, cle: str, valeur: Any) -> None:
        """Set data value."""
        self.donnees[cle] = valeur

    def get_donnee(self, cle: str, defaut: Any = None) -> Any:
        """Get data value."""
        return self.donnees.get(cle, defaut)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "nom": self.nom,
            "etat": self.etat.value,
            "niveau": self.niveau,
            "donnees": self.donnees,
            "historique": self.historique,
            "timestamp_creation": self.timestamp_creation,
        }


class Morphisme:
    """
    Morphism: transformation between objects.

    Represents a transformation f: A → B in the morphic universe.
    """

    def __init__(
        self,
        nom: str,
        fonction: Callable[[ObjetReactif], ObjetReactif],
        description: Optional[str] = None,
    ) -> None:
        """
        Initialize morphism.

        Args:
            nom: Morphism name
            fonction: Transformation function
            description: Optional description
        """
        self.nom = nom
        self.fonction = fonction
        self.description = description or f"Morphisme {nom}"
        self.execution_count = 0

    def appliquer(self, objet: ObjetReactif) -> ObjetReactif:
        """
        Apply morphism to object.

        Args:
            objet: Source object

        Returns:
            Transformed object
        """
        self.execution_count += 1
        return self.fonction(objet)


# ============================================================================
# U_G: Goal Universe - Objectives
# ============================================================================


class ObjectifG:
    """
    Goal/Objective in the Goal Universe U_G.

    Implements postulates G1-G5:
    - G1: An objective is a defined goal
    - G2: An objective is a boolean function
    - G3: Objectives are evaluable on objects
    - G4: Objectives guide fluxes
    - G5: Objectives are memorizable
    """

    def __init__(
        self,
        nom: str,
        condition: Callable[[ObjetReactif], bool],
        description: Optional[str] = None,
        priorite: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize objective.

        Args:
            nom: Objective name
            condition: Boolean condition function
            description: Optional description
            priorite: Priority level (higher = more important)
            metadata: Optional metadata
        """
        self.nom = nom
        self.condition = condition
        self.description = description or f"Objectif: {nom}"
        self.priorite = priorite
        self.metadata = metadata or {}
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        self.evaluation_count = 0
        self.success_count = 0

    def appliquer(self, objet: ObjetReactif) -> bool:
        """
        Evaluate objective on object (G3: evaluable).

        Args:
            objet: Object to evaluate

        Returns:
            True if objective is satisfied
        """
        self.evaluation_count += 1
        try:
            result = self.condition(objet)
            if result:
                self.success_count += 1
            return result
        except Exception as e:
            raise ObjectiveError(f"Failed to evaluate objective {self.nom}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (G5: memorizable)."""
        return {
            "id": self.id,
            "nom": self.nom,
            "description": self.description,
            "priorite": self.priorite,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "evaluation_count": self.evaluation_count,
            "success_count": self.success_count,
            "success_rate": (
                self.success_count / self.evaluation_count
                if self.evaluation_count > 0
                else 0
            ),
        }


class GenerateurObjectifs:
    """
    Objective Generator (G6: dynamically generatable).

    Dynamically creates objectives based on context.
    """

    def __init__(self) -> None:
        """Initialize objective generator."""
        self._templates: Dict[str, Callable] = {}

    def enregistrer_template(
        self, nom: str, generateur: Callable[[Dict[str, Any]], ObjectifG]
    ) -> None:
        """
        Register objective template.

        Args:
            nom: Template name
            generateur: Function that generates objective from context
        """
        self._templates[nom] = generateur

    def generer(self, template_nom: str, contexte: Dict[str, Any]) -> ObjectifG:
        """
        Generate objective from template.

        Args:
            template_nom: Name of template to use
            contexte: Context for generation

        Returns:
            Generated objective
        """
        if template_nom not in self._templates:
            raise ObjectiveError(f"Template '{template_nom}' not found")

        return self._templates[template_nom](contexte)


# ============================================================================
# U_Φ: Morphic Flux - Plan Generation and Execution
# ============================================================================


@dataclass
class EtapeFlux:
    """Step in a morphic flux."""

    nom: str
    action: Callable[[ObjetReactif], ObjetReactif]
    description: Optional[str] = None
    objectif_associe: Optional[ObjectifG] = None


class FluxPhi:
    """
    Morphic Flux - Plan generation and execution.

    Generates and executes plans to achieve objectives (G4: guide fluxes).
    """

    def __init__(
        self,
        nom: str,
        objectif: ObjectifG,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize morphic flux.

        Args:
            nom: Flux name
            objectif: Goal to achieve
            enable_logging: Enable detailed logging
        """
        self.nom = nom
        self.objectif = objectif
        self._etapes: List[EtapeFlux] = []
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self.id = str(uuid.uuid4())

    def ajouter_etape(
        self,
        nom: str,
        action: Callable[[ObjetReactif], ObjetReactif],
        description: Optional[str] = None,
    ) -> None:
        """
        Add step to flux.

        Args:
            nom: Step name
            action: Step action function
            description: Optional description
        """
        etape = EtapeFlux(nom=nom, action=action, description=description)
        self._etapes.append(etape)

        if self._enable_logging and self._logger:
            self._logger.debug(f"Added step '{nom}' to flux '{self.nom}'")

    def generer_plan(
        self, objet: ObjetReactif, strategie: str = "sequential"
    ) -> List[str]:
        """
        Generate execution plan.

        Args:
            objet: Object to plan for
            strategie: Planning strategy (sequential, parallel, adaptive)

        Returns:
            List of step names in execution order
        """
        # For now, simple sequential planning
        # Could be extended with AI-based planning
        plan = [etape.nom for etape in self._etapes]

        if self._enable_logging and self._logger:
            self._logger.info(f"Generated plan with {len(plan)} steps")

        return plan

    def executer(self, objet: ObjetReactif) -> Dict[str, Any]:
        """
        Execute flux on object.

        Args:
            objet: Object to transform

        Returns:
            Execution results
        """
        resultats = {
            "flux_id": self.id,
            "flux_nom": self.nom,
            "objet_initial": objet.nom,
            "etapes_executees": [],
            "objectif_atteint": False,
            "erreurs": [],
        }

        if self._enable_logging and self._logger:
            self._logger.info(f"Executing flux '{self.nom}' with {len(self._etapes)} steps")

        # Execute each step
        objet_courant = objet
        for i, etape in enumerate(self._etapes, 1):
            try:
                if self._enable_logging and self._logger:
                    self._logger.debug(f"Step {i}/{len(self._etapes)}: {etape.nom}")

                # Execute step
                objet_courant = etape.action(objet_courant)

                resultats["etapes_executees"].append(
                    {
                        "nom": etape.nom,
                        "status": "success",
                        "timestamp": time.time(),
                    }
                )

                # Check if objective is achieved
                if self.objectif.appliquer(objet_courant):
                    resultats["objectif_atteint"] = True
                    if self._enable_logging and self._logger:
                        self._logger.info(
                            f"Objective '{self.objectif.nom}' achieved after step {i}"
                        )
                    break

            except Exception as e:
                error_msg = f"Error in step {etape.nom}: {str(e)}"
                resultats["erreurs"].append(error_msg)

                if self._enable_logging and self._logger:
                    self._logger.error(error_msg)

                resultats["etapes_executees"].append(
                    {
                        "nom": etape.nom,
                        "status": "error",
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )

        resultats["objet_final"] = objet_courant.nom
        return resultats


# ============================================================================
# U_Ψ: Logical Selector - Filtering and Routing
# ============================================================================


class SelecteurPsi:
    """
    Logical Selector U_Ψ - Intelligent filtering and routing.

    Selects appropriate objects, objectives, or actions based on criteria.
    """

    def __init__(self, nom: str = "SelecteurPsi") -> None:
        """Initialize logical selector."""
        self.nom = nom
        self._regles: List[Tuple[Callable, Any]] = []

    def ajouter_regle(
        self, condition: Callable[[Any], bool], action_ou_valeur: Any
    ) -> None:
        """
        Add selection rule.

        Args:
            condition: Boolean condition function
            action_ou_valeur: Value or action to return if condition is true
        """
        self._regles.append((condition, action_ou_valeur))

    def selectionner(self, candidats: List[Any]) -> List[Any]:
        """
        Select candidates based on rules.

        Args:
            candidats: List of candidates to filter

        Returns:
            Filtered list
        """
        selectionnes = []

        for candidat in candidats:
            for condition, action_ou_valeur in self._regles:
                if condition(candidat):
                    selectionnes.append(candidat)
                    break

        return selectionnes

    def router(self, objet: Any) -> Optional[Any]:
        """
        Route object to appropriate action.

        Args:
            objet: Object to route

        Returns:
            Action or value for object
        """
        for condition, action_ou_valeur in self._regles:
            if condition(objet):
                return action_ou_valeur

        return None


# ============================================================================
# U_mem: Cognitive Memory - Traces and Vector Memory
# ============================================================================


@dataclass
class Trace:
    """
    Memory trace - record of an action or reasoning.

    Implements G5: objectives and actions are memorizable.
    """

    timestamp: float
    type_trace: str  # "action", "reasoning", "objective", "flux"
    contenu: Dict[str, Any]
    contexte: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type_trace": self.type_trace,
            "contenu": self.contenu,
            "contexte": self.contexte,
        }


class Memoire:
    """
    Cognitive memory system.

    Stores traces and enables replay of past experiences.
    """

    def __init__(self, nom: str = "Memoire") -> None:
        """Initialize memory system."""
        self.nom = nom
        self._traces: List[Trace] = []
        self._index_par_type: Dict[str, List[Trace]] = {}

    def enregistrer(
        self,
        type_trace: str,
        contenu: Dict[str, Any],
        contexte: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """
        Record trace in memory.

        Args:
            type_trace: Type of trace
            contenu: Trace content
            contexte: Optional context

        Returns:
            Created trace
        """
        trace = Trace(
            timestamp=time.time(),
            type_trace=type_trace,
            contenu=contenu,
            contexte=contexte or {},
        )

        self._traces.append(trace)

        # Index by type
        if type_trace not in self._index_par_type:
            self._index_par_type[type_trace] = []
        self._index_par_type[type_trace].append(trace)

        return trace

    def recuperer(
        self,
        type_trace: Optional[str] = None,
        filtre: Optional[Callable[[Trace], bool]] = None,
        limite: Optional[int] = None,
    ) -> List[Trace]:
        """
        Retrieve traces from memory.

        Args:
            type_trace: Optional type filter
            filtre: Optional custom filter function
            limite: Optional limit on number of traces

        Returns:
            List of matching traces
        """
        # Start with all traces or type-filtered traces
        if type_trace:
            traces = self._index_par_type.get(type_trace, [])
        else:
            traces = self._traces

        # Apply custom filter
        if filtre:
            traces = [t for t in traces if filtre(t)]

        # Apply limit
        if limite:
            traces = traces[-limite:]

        return traces

    def rejouer(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Replay a trace.

        Args:
            trace_id: ID of trace to replay

        Returns:
            Trace content if found
        """
        for trace in self._traces:
            if trace.id == trace_id:
                return trace.contenu

        return None

    def statistiques(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_traces": len(self._traces),
            "types": {
                type_: len(traces) for type_, traces in self._index_par_type.items()
            },
            "plus_ancienne": self._traces[0].timestamp if self._traces else None,
            "plus_recente": self._traces[-1].timestamp if self._traces else None,
        }


class MemoireVectorielle:
    """
    Vector-based cognitive memory.

    Integrates with KnowledgeLayer for semantic memory.
    """

    def __init__(
        self, memoire: Memoire, knowledge_store: Optional[Any] = None
    ) -> None:
        """
        Initialize vector memory.

        Args:
            memoire: Base memory system
            knowledge_store: Optional KnowledgeStore from KnowledgeLayer
        """
        self.memoire = memoire
        self.knowledge_store = knowledge_store

    def enregistrer_avec_embedding(
        self, type_trace: str, contenu: Dict[str, Any], texte: str
    ) -> Trace:
        """
        Record trace with vector embedding.

        Args:
            type_trace: Type of trace
            contenu: Trace content
            texte: Text for embedding

        Returns:
            Created trace
        """
        # Record in base memory
        trace = self.memoire.enregistrer(type_trace, contenu)

        # Add to vector store if available
        if self.knowledge_store:
            self.knowledge_store.add(
                text=texte, metadata={"trace_id": trace.id, "type": type_trace}
            )

        return trace

    def rechercher_similaire(self, requete: str, top_k: int = 5) -> List[Trace]:
        """
        Search for similar traces.

        Args:
            requete: Search query
            top_k: Number of results

        Returns:
            List of similar traces
        """
        if not self.knowledge_store:
            raise MemoryError("Vector store not configured")

        # Search in knowledge store
        results = self.knowledge_store.search(query=requete, top_k=top_k)

        # Retrieve corresponding traces
        traces = []
        for memory, score in results:
            trace_id = memory.metadata.get("trace_id")
            if trace_id:
                trace_contenu = self.memoire.rejouer(trace_id)
                if trace_contenu:
                    traces.append(trace_contenu)

        return traces


# ============================================================================
# RRLA: Reflection Reasoning Loop Agent
# ============================================================================


@dataclass
class EtapeRaisonnement:
    """Step in reasoning process."""

    etape: str  # "comprehension", "proposition", "evaluation", "reflection", "decision"
    contenu: str
    confiance: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ChainOfReasoning:
    """
    Chain of Reasoning (CoR) - Structured reasoning process.

    Implements multi-step reasoning with reflection.
    """

    def __init__(self, probleme: str) -> None:
        """
        Initialize reasoning chain.

        Args:
            probleme: Problem to reason about
        """
        self.probleme = probleme
        self._etapes: List[EtapeRaisonnement] = []
        self.id = str(uuid.uuid4())
        self.created_at = time.time()

    def ajouter_etape(
        self,
        etape: str,
        contenu: str,
        confiance: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add reasoning step.

        Args:
            etape: Step type
            contenu: Step content
            confiance: Confidence level (0-1)
            metadata: Optional metadata
        """
        etape_raisonnement = EtapeRaisonnement(
            etape=etape,
            contenu=contenu,
            confiance=confiance,
            metadata=metadata or {},
        )
        self._etapes.append(etape_raisonnement)

    def get_etapes(self) -> List[EtapeRaisonnement]:
        """Get all reasoning steps."""
        return self._etapes.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "probleme": self.probleme,
            "created_at": self.created_at,
            "etapes": [
                {
                    "etape": e.etape,
                    "contenu": e.contenu,
                    "confiance": e.confiance,
                    "metadata": e.metadata,
                    "timestamp": e.timestamp,
                }
                for e in self._etapes
            ],
            "confiance_moyenne": (
                sum(e.confiance for e in self._etapes) / len(self._etapes)
                if self._etapes
                else 0
            ),
        }


class RRLA:
    """
    Reflection Reasoning Loop Agent.

    Implements structured reasoning with:
    - Comprehension: Understand the problem
    - Proposition: Generate solutions
    - Evaluation: Assess solutions
    - Reflection: Reflect on process
    - Decision: Make final decision
    """

    def __init__(self, memoire: Memoire, enable_logging: bool = False) -> None:
        """
        Initialize RRLA.

        Args:
            memoire: Memory system for storing reasoning
            enable_logging: Enable detailed logging
        """
        self.memoire = memoire
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self._raisonnements: List[ChainOfReasoning] = []

    def raisonner(
        self,
        probleme: str,
        contexte: Optional[Dict[str, Any]] = None,
        mode: str = "complet",
    ) -> ChainOfReasoning:
        """
        Perform reasoning on problem.

        Args:
            probleme: Problem to reason about
            contexte: Optional context
            mode: Reasoning mode (complet, rapide, approfondi)

        Returns:
            Chain of reasoning
        """
        cor = ChainOfReasoning(probleme=probleme)
        self._raisonnements.append(cor)

        if self._enable_logging and self._logger:
            self._logger.info(f"Starting reasoning on: {probleme}")

        # Step 1: Comprehension
        comprehension = self._comprendre(probleme, contexte)
        cor.ajouter_etape("comprehension", comprehension, confiance=0.9)

        # Step 2: Proposition
        propositions = self._proposer(probleme, contexte)
        cor.ajouter_etape("proposition", propositions, confiance=0.8)

        # Step 3: Evaluation
        if mode in ["complet", "approfondi"]:
            evaluation = self._evaluer(propositions, contexte)
            cor.ajouter_etape("evaluation", evaluation, confiance=0.85)

        # Step 4: Reflection
        if mode == "approfondi":
            reflection = self._reflechir(cor.get_etapes())
            cor.ajouter_etape("reflection", reflection, confiance=0.9)

        # Step 5: Decision
        decision = self._decider(cor.get_etapes())
        cor.ajouter_etape("decision", decision, confiance=0.95)

        # Store in memory
        self.memoire.enregistrer(
            type_trace="reasoning",
            contenu=cor.to_dict(),
            contexte=contexte or {},
        )

        return cor

    def _comprendre(
        self, probleme: str, contexte: Optional[Dict[str, Any]]
    ) -> str:
        """Comprehension phase."""
        # Simplified - in real implementation, would use LLM
        return f"Analyzing problem: {probleme}"

    def _proposer(self, probleme: str, contexte: Optional[Dict[str, Any]]) -> str:
        """Proposition phase."""
        # Simplified - in real implementation, would generate multiple solutions
        return f"Proposed solutions for: {probleme}"

    def _evaluer(self, propositions: str, contexte: Optional[Dict[str, Any]]) -> str:
        """Evaluation phase."""
        return f"Evaluating proposed solutions"

    def _reflechir(self, etapes: List[EtapeRaisonnement]) -> str:
        """Reflection phase."""
        return f"Reflecting on reasoning process with {len(etapes)} steps"

    def _decider(self, etapes: List[EtapeRaisonnement]) -> str:
        """Decision phase."""
        return f"Final decision based on {len(etapes)} reasoning steps"


# ============================================================================
# Persona: Agent Personality System (Nümtema)
# ============================================================================


@dataclass
class Persona:
    """
    Agent persona definition.

    Defines personality, style, and behavior patterns.
    """

    nom: str
    description: str
    traits: List[str] = field(default_factory=list)
    style_communication: str = "professionnel"
    niveau_detail: str = "moyen"  # minimal, moyen, detaille
    mode_conseil: str = "equilibre"  # directif, equilibre, socratique
    superpouvoirs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nom": self.nom,
            "description": self.description,
            "traits": self.traits,
            "style_communication": self.style_communication,
            "niveau_detail": self.niveau_detail,
            "mode_conseil": self.mode_conseil,
            "superpouvoirs": self.superpouvoirs,
            "metadata": self.metadata,
        }


# Predefined Nümtema persona
NUMTEMA_PERSONA = Persona(
    nom="Nümtema",
    description="Assistant bienveillant, expert en raisonnement morphique",
    traits=[
        "bienveillant",
        "expert",
        "raisonneur",
        "pedagogique",
        "adaptatif",
    ],
    style_communication="professionnel_amical",
    niveau_detail="adaptatif",
    mode_conseil="equilibre",
    superpouvoirs=[
        "raisonnement_guide",
        "replay_memoire",
        "visualisation_CoR",
        "adaptation_contexte",
        "generation_objectifs",
    ],
)


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class ObjectifEvaluationNode(Node):
    """
    Node that evaluates an objective on an object.

    Integrates U_G with Flow system.
    """

    def __init__(
        self,
        objectif: ObjectifG,
        objet_key: str = "objet_reactif",
        output_key: str = "objectif_atteint",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize objective evaluation node.

        Args:
            objectif: Objective to evaluate
            objet_key: Key in SharedStore to read object from
            output_key: Key in SharedStore to write result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(
            name or f"EvalObjectif_{objectif.nom}", enable_logging=enable_logging
        )
        self.objectif = objectif
        self.objet_key = objet_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute objective evaluation.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get object
        objet = shared.get_required(self.objet_key)

        # Evaluate objective
        atteint = self.objectif.appliquer(objet)

        # Store result
        shared.set(self.output_key, atteint)

        return {
            "objectif": self.objectif.nom,
            "atteint": atteint,
            "evaluation_count": self.objectif.evaluation_count,
        }


class FluxExecutionNode(Node):
    """
    Node that executes a morphic flux.

    Integrates U_Φ with Flow system.
    """

    def __init__(
        self,
        flux: FluxPhi,
        objet_key: str = "objet_reactif",
        output_key: str = "flux_result",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize flux execution node.

        Args:
            flux: Flux to execute
            objet_key: Key in SharedStore to read object from
            output_key: Key in SharedStore to write result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(
            name or f"ExecFlux_{flux.nom}", enable_logging=enable_logging
        )
        self.flux = flux
        self.objet_key = objet_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute flux.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get object
        objet = shared.get_required(self.objet_key)

        # Execute flux
        result = self.flux.executer(objet)

        # Store result
        shared.set(self.output_key, result)

        return result


class ReasoningNode(Node):
    """
    Node that performs reasoning using RRLA.

    Integrates RRLA with Flow system.
    """

    def __init__(
        self,
        rrla: RRLA,
        probleme_key: str = "probleme",
        contexte_key: Optional[str] = None,
        output_key: str = "reasoning_result",
        mode: str = "complet",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize reasoning node.

        Args:
            rrla: RRLA instance
            probleme_key: Key in SharedStore to read problem from
            contexte_key: Optional key to read context from
            output_key: Key in SharedStore to write result to
            mode: Reasoning mode
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Reasoning", enable_logging=enable_logging)
        self.rrla = rrla
        self.probleme_key = probleme_key
        self.contexte_key = contexte_key
        self.output_key = output_key
        self.mode = mode

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute reasoning.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get problem
        probleme = shared.get_required(self.probleme_key)

        # Get optional context
        contexte = None
        if self.contexte_key:
            contexte = shared.get(self.contexte_key)

        # Perform reasoning
        cor = self.rrla.raisonner(probleme, contexte, mode=self.mode)

        # Store result
        shared.set(self.output_key, cor.to_dict())

        return {
            "reasoning_id": cor.id,
            "etapes_count": len(cor.get_etapes()),
            "confiance_moyenne": cor.to_dict()["confiance_moyenne"],
        }


class MemoireNode(Node):
    """
    Node that records trace in memory.

    Integrates U_mem with Flow system.
    """

    def __init__(
        self,
        memoire: Memoire,
        type_trace: str,
        contenu_key: str = "trace_contenu",
        contexte_key: Optional[str] = None,
        output_key: str = "trace_id",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize memory node.

        Args:
            memoire: Memory system
            type_trace: Type of trace to record
            contenu_key: Key in SharedStore to read content from
            contexte_key: Optional key to read context from
            output_key: Key in SharedStore to write trace ID to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Memoire", enable_logging=enable_logging)
        self.memoire = memoire
        self.type_trace = type_trace
        self.contenu_key = contenu_key
        self.contexte_key = contexte_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Record trace in memory.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get content
        contenu = shared.get_required(self.contenu_key)

        # Get optional context
        contexte = None
        if self.contexte_key:
            contexte = shared.get(self.contexte_key)

        # Record trace
        trace = self.memoire.enregistrer(self.type_trace, contenu, contexte)

        # Store trace ID
        shared.set(self.output_key, trace.id)

        return {
            "trace_id": trace.id,
            "type_trace": self.type_trace,
            "timestamp": trace.timestamp,
        }
