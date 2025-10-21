"""
Tests for MorphicLayer

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import pytest
from num_agents import (
    # U₀
    ObjetReactif,
    EtatObjet,
    Morphisme,
    # U_G
    ObjectifG,
    GenerateurObjectifs,
    # U_Φ
    FluxPhi,
    # U_Ψ
    SelecteurPsi,
    # U_mem
    Trace,
    Memoire,
    # RRLA
    ChainOfReasoning,
    RRLA,
    # Persona
    Persona,
    NUMTEMA_PERSONA,
    # Nodes
    ObjectifEvaluationNode,
    FluxExecutionNode,
    ReasoningNode,
    MemoireNode,
    Flow,
    SharedStore,
)


# ============================================================================
# Test U₀: Base Morphic Universe
# ============================================================================


def test_objet_reactif_creation():
    """Test creating reactive object."""
    obj = ObjetReactif(nom="test_obj", niveau=1)

    assert obj.nom == "test_obj"
    assert obj.etat == EtatObjet.INACTIF
    assert obj.niveau == 1
    assert len(obj.historique) == 0


def test_objet_reactif_change_state():
    """Test changing object state."""
    obj = ObjetReactif(nom="obj")

    obj.changer_etat(EtatObjet.ACTIF, raison="Activation")

    assert obj.etat == EtatObjet.ACTIF
    assert len(obj.historique) == 1
    assert obj.historique[0]["nouvel_etat"] == "actif"
    assert obj.historique[0]["raison"] == "Activation"


def test_objet_reactif_data():
    """Test object data management."""
    obj = ObjetReactif(nom="obj")

    obj.set_donnee("temperature", 25)
    obj.set_donnee("humidity", 60)

    assert obj.get_donnee("temperature") == 25
    assert obj.get_donnee("humidity") == 60
    assert obj.get_donnee("pressure", 1013) == 1013  # Default


def test_morphisme():
    """Test morphism application."""

    def transformer(obj: ObjetReactif) -> ObjetReactif:
        obj.incrementer_niveau()
        obj.changer_etat(EtatObjet.ACTIF)
        return obj

    morphisme = Morphisme(nom="activation", fonction=transformer)

    obj = ObjetReactif(nom="obj", niveau=0)
    obj_transforme = morphisme.appliquer(obj)

    assert obj_transforme.niveau == 1
    assert obj_transforme.etat == EtatObjet.ACTIF
    assert morphisme.execution_count == 1


# ============================================================================
# Test U_G: Goal Universe
# ============================================================================


def test_objectif_creation():
    """Test creating objective."""

    def condition(obj: ObjetReactif) -> bool:
        return obj.niveau >= 5

    objectif = ObjectifG(
        nom="atteindre_niveau_5", condition=condition, priorite=2
    )

    assert objectif.nom == "atteindre_niveau_5"
    assert objectif.priorite == 2
    assert objectif.evaluation_count == 0


def test_objectif_evaluation():
    """Test objective evaluation (G3)."""

    def condition(obj: ObjetReactif) -> bool:
        return obj.niveau >= 5

    objectif = ObjectifG(nom="niveau_5", condition=condition)

    obj1 = ObjetReactif(nom="obj1", niveau=3)
    obj2 = ObjetReactif(nom="obj2", niveau=7)

    assert objectif.appliquer(obj1) is False
    assert objectif.appliquer(obj2) is True
    assert objectif.evaluation_count == 2
    assert objectif.success_count == 1


def test_objectif_to_dict():
    """Test objective serialization (G5)."""

    def condition(obj: ObjetReactif) -> bool:
        return obj.etat == EtatObjet.TERMINE

    objectif = ObjectifG(nom="termine", condition=condition)

    obj = ObjetReactif(nom="obj")
    obj.changer_etat(EtatObjet.TERMINE)
    objectif.appliquer(obj)

    data = objectif.to_dict()

    assert data["nom"] == "termine"
    assert data["evaluation_count"] == 1
    assert data["success_rate"] == 1.0


def test_generateur_objectifs():
    """Test objective generator (G6)."""
    generateur = GenerateurObjectifs()

    # Register template
    def generer_objectif_niveau(contexte):
        niveau_cible = contexte["niveau"]

        def condition(obj):
            return obj.niveau >= niveau_cible

        return ObjectifG(
            nom=f"niveau_{niveau_cible}",
            condition=condition,
            description=f"Atteindre niveau {niveau_cible}",
        )

    generateur.enregistrer_template("niveau", generer_objectif_niveau)

    # Generate objective
    objectif = generateur.generer("niveau", {"niveau": 10})

    assert objectif.nom == "niveau_10"

    # Test generated objective
    obj = ObjetReactif(nom="obj", niveau=12)
    assert objectif.appliquer(obj) is True


# ============================================================================
# Test U_Φ: Morphic Flux
# ============================================================================


def test_flux_phi_creation():
    """Test creating morphic flux."""

    def condition(obj):
        return obj.niveau >= 3

    objectif = ObjectifG(nom="niveau_3", condition=condition)
    flux = FluxPhi(nom="test_flux", objectif=objectif)

    assert flux.nom == "test_flux"
    assert flux.objectif == objectif


def test_flux_phi_add_steps():
    """Test adding steps to flux."""

    def condition(obj):
        return obj.niveau >= 2

    objectif = ObjectifG(nom="niveau_2", condition=condition)
    flux = FluxPhi(nom="flux", objectif=objectif)

    def etape1(obj):
        obj.incrementer_niveau()
        return obj

    def etape2(obj):
        obj.incrementer_niveau()
        return obj

    flux.ajouter_etape("incrementer1", etape1)
    flux.ajouter_etape("incrementer2", etape2)

    plan = flux.generer_plan(ObjetReactif(nom="obj"))
    assert len(plan) == 2
    assert plan == ["incrementer1", "incrementer2"]


def test_flux_phi_execution():
    """Test flux execution (G4: guide fluxes)."""

    def condition(obj):
        return obj.niveau >= 3

    objectif = ObjectifG(nom="niveau_3", condition=condition)
    flux = FluxPhi(nom="flux", objectif=objectif)

    # Add steps
    def incrementer(obj):
        obj.incrementer_niveau()
        return obj

    flux.ajouter_etape("etape1", incrementer)
    flux.ajouter_etape("etape2", incrementer)
    flux.ajouter_etape("etape3", incrementer)
    flux.ajouter_etape("etape4", incrementer)  # Won't execute (objective met)

    # Execute
    obj = ObjetReactif(nom="obj", niveau=0)
    result = flux.executer(obj)

    assert result["objectif_atteint"] is True
    assert len(result["etapes_executees"]) == 3  # Stops after objective met


# ============================================================================
# Test U_Ψ: Logical Selector
# ============================================================================


def test_selecteur_psi_selection():
    """Test logical selector filtering."""
    selecteur = SelecteurPsi()

    # Add rules
    selecteur.ajouter_regle(lambda x: x > 5, "grand")
    selecteur.ajouter_regle(lambda x: x < 3, "petit")

    candidats = [1, 2, 3, 4, 5, 6, 7, 8]
    selectionnes = selecteur.selectionner(candidats)

    # Should select 1, 2, 6, 7, 8 (< 3 or > 5)
    assert 1 in selectionnes
    assert 2 in selectionnes
    assert 6 in selectionnes
    assert 3 not in selectionnes
    assert 4 not in selectionnes


def test_selecteur_psi_routing():
    """Test logical selector routing."""
    selecteur = SelecteurPsi()

    # Add routing rules
    selecteur.ajouter_regle(lambda x: x["type"] == "urgent", "handle_urgent")
    selecteur.ajouter_regle(lambda x: x["type"] == "normal", "handle_normal")

    urgent = {"type": "urgent", "data": "urgent task"}
    normal = {"type": "normal", "data": "normal task"}

    assert selecteur.router(urgent) == "handle_urgent"
    assert selecteur.router(normal) == "handle_normal"


# ============================================================================
# Test U_mem: Cognitive Memory
# ============================================================================


def test_trace_creation():
    """Test creating memory trace."""
    trace = Trace(
        timestamp=1234567890.0,
        type_trace="action",
        contenu={"action": "test"},
        contexte={"user": "alice"},
    )

    assert trace.type_trace == "action"
    assert trace.contenu["action"] == "test"
    assert trace.contexte["user"] == "alice"


def test_memoire_enregistrer():
    """Test recording traces in memory."""
    memoire = Memoire()

    trace1 = memoire.enregistrer(
        "action", {"action": "login"}, {"user": "alice"}
    )
    trace2 = memoire.enregistrer("reasoning", {"steps": 5})

    assert trace1.type_trace == "action"
    assert trace2.type_trace == "reasoning"


def test_memoire_recuperer():
    """Test retrieving traces from memory."""
    memoire = Memoire()

    memoire.enregistrer("action", {"action": "login"})
    memoire.enregistrer("action", {"action": "logout"})
    memoire.enregistrer("reasoning", {"steps": 5})

    # Retrieve all
    all_traces = memoire.recuperer()
    assert len(all_traces) == 3

    # Retrieve by type
    action_traces = memoire.recuperer(type_trace="action")
    assert len(action_traces) == 2

    # Retrieve with limit
    limited = memoire.recuperer(limite=2)
    assert len(limited) == 2


def test_memoire_rejouer():
    """Test replaying traces."""
    memoire = Memoire()

    trace = memoire.enregistrer("test", {"value": 42})
    rejouee = memoire.rejouer(trace.id)

    assert rejouee["value"] == 42


def test_memoire_statistiques():
    """Test memory statistics."""
    memoire = Memoire()

    memoire.enregistrer("action", {})
    memoire.enregistrer("action", {})
    memoire.enregistrer("reasoning", {})

    stats = memoire.statistiques()

    assert stats["total_traces"] == 3
    assert stats["types"]["action"] == 2
    assert stats["types"]["reasoning"] == 1


# ============================================================================
# Test RRLA: Reflection Reasoning Loop Agent
# ============================================================================


def test_chain_of_reasoning():
    """Test chain of reasoning."""
    cor = ChainOfReasoning(probleme="How to optimize code?")

    cor.ajouter_etape("comprehension", "Understanding the problem", confiance=0.9)
    cor.ajouter_etape("proposition", "Proposed solution A", confiance=0.8)
    cor.ajouter_etape("decision", "Choose solution A", confiance=0.95)

    etapes = cor.get_etapes()
    assert len(etapes) == 3
    assert etapes[0].etape == "comprehension"
    assert etapes[1].confiance == 0.8


def test_chain_of_reasoning_to_dict():
    """Test CoR serialization."""
    cor = ChainOfReasoning(probleme="Test problem")

    cor.ajouter_etape("step1", "Content 1", confiance=0.9)
    cor.ajouter_etape("step2", "Content 2", confiance=0.8)

    data = cor.to_dict()

    assert data["probleme"] == "Test problem"
    assert len(data["etapes"]) == 2
    assert data["confiance_moyenne"] == 0.85  # (0.9 + 0.8) / 2


def test_rrla_raisonner():
    """Test RRLA reasoning."""
    memoire = Memoire()
    rrla = RRLA(memoire=memoire)

    cor = rrla.raisonner(
        probleme="How to improve performance?",
        contexte={"current_perf": "slow"},
        mode="complet",
    )

    etapes = cor.get_etapes()

    # Should have comprehension, proposition, evaluation, decision
    assert len(etapes) >= 4
    etape_types = [e.etape for e in etapes]
    assert "comprehension" in etape_types
    assert "proposition" in etape_types
    assert "decision" in etape_types


def test_rrla_memory_storage():
    """Test that reasoning is stored in memory."""
    memoire = Memoire()
    rrla = RRLA(memoire=memoire)

    rrla.raisonner("Problem 1")
    rrla.raisonner("Problem 2")

    # Check memory
    reasoning_traces = memoire.recuperer(type_trace="reasoning")
    assert len(reasoning_traces) == 2


# ============================================================================
# Test Persona
# ============================================================================


def test_persona_creation():
    """Test creating persona."""
    persona = Persona(
        nom="TestBot",
        description="A test bot",
        traits=["helpful", "friendly"],
        style_communication="casual",
    )

    assert persona.nom == "TestBot"
    assert "helpful" in persona.traits


def test_numtema_persona():
    """Test predefined Nümtema persona."""
    assert NUMTEMA_PERSONA.nom == "Nümtema"
    assert "bienveillant" in NUMTEMA_PERSONA.traits
    assert "raisonnement_guide" in NUMTEMA_PERSONA.superpouvoirs


def test_persona_to_dict():
    """Test persona serialization."""
    persona = Persona(nom="Bot", description="Test")
    data = persona.to_dict()

    assert data["nom"] == "Bot"
    assert "style_communication" in data


# ============================================================================
# Test Nodes (Flow Integration)
# ============================================================================


def test_objectif_evaluation_node():
    """Test ObjectifEvaluationNode."""

    def condition(obj):
        return obj.niveau >= 5

    objectif = ObjectifG(nom="niveau_5", condition=condition)
    node = ObjectifEvaluationNode(objectif=objectif)

    shared = SharedStore()
    obj = ObjetReactif(nom="obj", niveau=7)
    shared.set("objet_reactif", obj)

    result = node.exec(shared)

    assert result["atteint"] is True
    assert shared.get("objectif_atteint") is True


def test_flux_execution_node():
    """Test FluxExecutionNode."""

    def condition(obj):
        return obj.niveau >= 2

    objectif = ObjectifG(nom="niveau_2", condition=condition)
    flux = FluxPhi(nom="flux", objectif=objectif)

    def incrementer(obj):
        obj.incrementer_niveau()
        return obj

    flux.ajouter_etape("inc1", incrementer)
    flux.ajouter_etape("inc2", incrementer)

    node = FluxExecutionNode(flux=flux)

    shared = SharedStore()
    obj = ObjetReactif(nom="obj", niveau=0)
    shared.set("objet_reactif", obj)

    result = node.exec(shared)

    assert result["objectif_atteint"] is True


def test_reasoning_node():
    """Test ReasoningNode."""
    memoire = Memoire()
    rrla = RRLA(memoire=memoire)
    node = ReasoningNode(rrla=rrla, mode="rapide")

    shared = SharedStore()
    shared.set("probleme", "How to solve this?")

    result = node.exec(shared)

    assert "reasoning_id" in result
    assert result["etapes_count"] > 0
    assert "confiance_moyenne" in result


def test_memoire_node():
    """Test MemoireNode."""
    memoire = Memoire()
    node = MemoireNode(memoire=memoire, type_trace="test_trace")

    shared = SharedStore()
    shared.set("trace_contenu", {"data": "test"})

    result = node.exec(shared)

    assert "trace_id" in result
    assert result["type_trace"] == "test_trace"

    # Verify in memory
    traces = memoire.recuperer(type_trace="test_trace")
    assert len(traces) == 1


# ============================================================================
# Integration Tests
# ============================================================================


def test_morphic_layer_integration():
    """Integration test: Full morphic workflow."""

    # 1. Create objective (U_G)
    def condition(obj):
        return obj.niveau >= 5 and obj.etat == EtatObjet.TERMINE

    objectif = ObjectifG(
        nom="objectif_complet",
        condition=condition,
        description="Atteindre niveau 5 et état TERMINE",
    )

    # 2. Create flux (U_Φ)
    flux = FluxPhi(nom="flux_progression", objectif=objectif)

    def incrementer_niveau(obj):
        obj.incrementer_niveau()
        return obj

    def terminer(obj):
        obj.changer_etat(EtatObjet.TERMINE)
        return obj

    flux.ajouter_etape("inc1", incrementer_niveau)
    flux.ajouter_etape("inc2", incrementer_niveau)
    flux.ajouter_etape("inc3", incrementer_niveau)
    flux.ajouter_etape("inc4", incrementer_niveau)
    flux.ajouter_etape("inc5", incrementer_niveau)
    flux.ajouter_etape("finaliser", terminer)

    # 3. Create object (U₀)
    obj = ObjetReactif(nom="agent", niveau=0)

    # 4. Setup memory (U_mem)
    memoire = Memoire()

    # 5. Execute flux
    result = flux.executer(obj)

    assert result["objectif_atteint"] is True

    # 6. Store in memory
    memoire.enregistrer(
        type_trace="flux_execution",
        contenu=result,
        contexte={"objectif": objectif.nom},
    )

    # 7. Verify memory
    flux_traces = memoire.recuperer(type_trace="flux_execution")
    assert len(flux_traces) == 1

    # 8. Reasoning on result
    rrla = RRLA(memoire=memoire)
    cor = rrla.raisonner(
        probleme="Analyser l'exécution du flux",
        contexte={"flux_result": result},
    )

    assert len(cor.get_etapes()) > 0


def test_flow_integration():
    """Test MorphicLayer integration with Flow."""

    # Setup
    def condition(obj):
        return obj.niveau >= 3

    objectif = ObjectifG(nom="niveau_3", condition=condition)
    flux = FluxPhi(nom="flux", objectif=objectif)

    def incrementer(obj):
        obj.incrementer_niveau()
        return obj

    flux.ajouter_etape("inc1", incrementer)
    flux.ajouter_etape("inc2", incrementer)
    flux.ajouter_etape("inc3", incrementer)

    memoire = Memoire()

    # Build flow
    flow = Flow(name="MorphicFlow")

    # Create nodes
    flux_node = FluxExecutionNode(flux=flux, name="ExecuteFlux")
    eval_node = ObjectifEvaluationNode(objectif=objectif, name="EvalObjectif")
    memory_node = MemoireNode(
        memoire=memoire, type_trace="flux", contenu_key="flux_result", name="StoreMemory"
    )

    flow.add_node(flux_node)
    flow.add_node(eval_node)
    flow.add_node(memory_node)
    flow.set_start(flux_node)

    # Execute
    obj = ObjetReactif(nom="agent", niveau=0)
    results = flow.execute(initial_data={"objet_reactif": obj})

    # Verify
    assert flow._shared.get("objectif_atteint") is True
    assert len(memoire.recuperer()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
