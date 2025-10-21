"""
MorphicLayer Demo - Demonstrating the Morphic Universe System

This comprehensive demo shows:
1. U₀: Reactive objects and morphisms
2. U_G: Goal-oriented objectives (Postulates G1-G6)
3. U_Φ: Morphic flux for plan generation
4. U_Ψ: Logical selector for intelligent routing
5. U_mem: Cognitive memory with traces
6. RRLA: Reflection Reasoning Loop Agent
7. CoR: Chain of Reasoning
8. Persona: Agent personality (Nümtema)
9. Flow Integration

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

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
    Memoire,
    MemoireVectorielle,
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
    Node,
    SharedStore,
)


# ============================================================================
# Example 1: U₀ - Base Morphic Universe
# ============================================================================


def demo_univers_morphique():
    """Demonstrate reactive objects and morphisms."""
    print("=" * 70)
    print("Example 1: U₀ - Univers Morphique (Reactive Objects)")
    print("=" * 70)

    # Create reactive object
    agent = ObjetReactif(nom="Agent_Alpha", niveau=0)
    print(f"\n🤖 Created agent: {agent.nom}")
    print(f"   Initial state: {agent.etat.value}, Level: {agent.niveau}")

    # Change states
    agent.changer_etat(EtatObjet.ACTIF, raison="Agent démarré")
    print(f"\n✨ State changed to: {agent.etat.value}")

    agent.set_donnee("mission", "Analyser les données")
    agent.set_donnee("priority", "high")
    print(f"📊 Data set: {agent.donnees}")

    # Define morphisms (transformations)
    def evoluer(obj: ObjetReactif) -> ObjetReactif:
        """Morphism: evolve agent to next level."""
        obj.incrementer_niveau()
        obj.changer_etat(EtatObjet.EN_COURS, raison="Evolution")
        return obj

    def terminer_mission(obj: ObjetReactif) -> ObjetReactif:
        """Morphism: complete mission."""
        obj.changer_etat(EtatObjet.TERMINE, raison="Mission accomplie")
        obj.set_donnee("completed_at", "2025-10-21")
        return obj

    # Create morphisms
    morphisme_evolution = Morphisme(nom="Evoluer", fonction=evoluer)
    morphisme_terminer = Morphisme(nom="Terminer", fonction=terminer_mission)

    print(f"\n🔄 Applying morphism: {morphisme_evolution.nom}")
    agent = morphisme_evolution.appliquer(agent)
    print(f"   New level: {agent.niveau}, State: {agent.etat.value}")

    print(f"\n🔄 Applying morphism: {morphisme_terminer.nom}")
    agent = morphisme_terminer.appliquer(agent)
    print(f"   Final state: {agent.etat.value}")

    # Show history
    print(f"\n📜 Agent History ({len(agent.historique)} events):")
    for event in agent.historique[-3:]:
        print(f"   - {event['nouvel_etat']}: {event['raison']}")


# ============================================================================
# Example 2: U_G - Goal Universe (Postulates G1-G6)
# ============================================================================


def demo_objectifs():
    """Demonstrate goal-oriented objectives."""
    print("\n" + "=" * 70)
    print("Example 2: U_G - Univers des Objectifs (Postulates G1-G6)")
    print("=" * 70)

    # G1: An objective is a defined goal
    # G2: An objective is a boolean function
    print("\n📌 Postulate G1 & G2: Objective = Defined Boolean Goal")

    def atteindre_niveau_expert(obj: ObjetReactif) -> bool:
        return obj.niveau >= 10

    objectif_expert = ObjectifG(
        nom="Devenir Expert",
        condition=atteindre_niveau_expert,
        description="Atteindre le niveau 10 d'expertise",
        priorite=5,
    )

    print(f"   Created: {objectif_expert.nom}")
    print(f"   Priority: {objectif_expert.priorite}")

    # G3: Objectives are evaluable on objects
    print("\n✅ Postulate G3: Objectives are Evaluable")

    agent_novice = ObjetReactif(nom="Novice", niveau=3)
    agent_expert = ObjetReactif(nom="Expert", niveau=12)

    result_novice = objectif_expert.appliquer(agent_novice)
    result_expert = objectif_expert.appliquer(agent_expert)

    print(f"   Novice (level {agent_novice.niveau}): {result_novice}")
    print(f"   Expert (level {agent_expert.niveau}): {result_expert}")

    # G5: Objectives are memorizable
    print("\n💾 Postulate G5: Objectives are Memorizable")
    obj_data = objectif_expert.to_dict()
    print(f"   Serialized: {obj_data['nom']}")
    print(f"   Success rate: {obj_data['success_rate']:.0%}")

    # G6: Objectives are dynamically generatable
    print("\n🎯 Postulate G6: Dynamic Objective Generation")

    generateur = GenerateurObjectifs()

    # Register template
    def generer_objectif_donnee(contexte):
        cle = contexte["cle"]
        valeur_attendue = contexte["valeur"]

        def condition(obj):
            return obj.get_donnee(cle) == valeur_attendue

        return ObjectifG(
            nom=f"Objectif_{cle}_{valeur_attendue}",
            condition=condition,
            description=f"Vérifier que {cle} = {valeur_attendue}",
        )

    generateur.enregistrer_template("donnee", generer_objectif_donnee)

    # Generate objective
    objectif_genere = generateur.generer(
        "donnee", {"cle": "status", "valeur": "active"}
    )

    print(f"   Generated: {objectif_genere.nom}")

    # Test generated objective
    agent = ObjetReactif(nom="Agent")
    agent.set_donnee("status", "active")
    print(f"   Evaluation: {objectif_genere.appliquer(agent)}")


# ============================================================================
# Example 3: U_Φ - Morphic Flux (G4: Guide Fluxes)
# ============================================================================


def demo_flux_morphique():
    """Demonstrate morphic flux and plan execution."""
    print("\n" + "=" * 70)
    print("Example 3: U_Φ - Flux Morphique (G4: Objectives Guide Fluxes)")
    print("=" * 70)

    # Define objective
    def objectif_agent_operationnel(obj: ObjetReactif) -> bool:
        return (
            obj.niveau >= 5
            and obj.etat == EtatObjet.ACTIF
            and obj.get_donnee("trained") is True
        )

    objectif = ObjectifG(
        nom="Agent Opérationnel", condition=objectif_agent_operationnel
    )

    print(f"\n🎯 Objective: {objectif.nom}")

    # Create flux
    flux = FluxPhi(nom="Formation Agent", objectif=objectif, enable_logging=False)

    # Define steps
    def initialiser(obj):
        obj.changer_etat(EtatObjet.EN_COURS, raison="Initialisation")
        obj.set_donnee("initialized", True)
        return obj

    def entrainer(obj):
        for i in range(5):
            obj.incrementer_niveau()
        obj.set_donnee("trained", True)
        return obj

    def activer(obj):
        obj.changer_etat(EtatObjet.ACTIF, raison="Formation terminée")
        return obj

    # Add steps to flux
    flux.ajouter_etape("Initialiser", initialiser, "Initialize agent")
    flux.ajouter_etape("Entraîner", entrainer, "Train agent to level 5")
    flux.ajouter_etape("Activer", activer, "Activate agent")

    print(f"\n📋 Flux: {flux.nom}")

    # Generate plan
    agent = ObjetReactif(nom="Agent_Beta", niveau=0)
    plan = flux.generer_plan(agent)
    print(f"   Generated plan: {' → '.join(plan)}")

    # Execute flux
    print(f"\n🚀 Executing flux...")
    result = flux.executer(agent)

    print(f"\n✅ Results:")
    print(f"   Objective achieved: {result['objectif_atteint']}")
    print(f"   Steps executed: {len(result['etapes_executees'])}")
    print(f"   Final agent level: {agent.niveau}")
    print(f"   Final state: {agent.etat.value}")


# ============================================================================
# Example 4: U_Ψ - Logical Selector
# ============================================================================


def demo_selecteur_logique():
    """Demonstrate logical selector for routing."""
    print("\n" + "=" * 70)
    print("Example 4: U_Ψ - Sélecteur Logique (Intelligent Routing)")
    print("=" * 70)

    selecteur = SelecteurPsi(nom="Routeur de Tâches")

    # Define routing rules
    print("\n📡 Setting up routing rules...")

    selecteur.ajouter_regle(
        condition=lambda task: task.get("priority") == "urgent", action_ou_valeur="handle_urgent"
    )

    selecteur.ajouter_regle(
        condition=lambda task: task.get("type") == "analysis", action_ou_valeur="analyze_data"
    )

    selecteur.ajouter_regle(
        condition=lambda task: task.get("type") == "reporting", action_ou_valeur="generate_report"
    )

    # Test routing
    tasks = [
        {"id": 1, "type": "analysis", "priority": "normal", "name": "Analyze sales"},
        {"id": 2, "type": "reporting", "priority": "urgent", "name": "Q4 Report"},
        {"id": 3, "type": "analysis", "priority": "urgent", "name": "Urgent analysis"},
    ]

    print(f"\n🎯 Routing {len(tasks)} tasks:")
    for task in tasks:
        route = selecteur.router(task)
        print(f"   Task {task['id']} ({task['name']}) → {route}")

    # Selection example
    print(f"\n🔍 Selecting high-priority agents:")

    agents = [
        ObjetReactif(nom="Agent_A", niveau=3),
        ObjetReactif(nom="Agent_B", niveau=8),
        ObjetReactif(nom="Agent_C", niveau=5),
    ]

    selecteur_agents = SelecteurPsi()
    selecteur_agents.ajouter_regle(lambda a: a.niveau >= 5, "qualified")

    selectionnes = selecteur_agents.selectionner(agents)
    print(f"   Qualified agents: {[a.nom for a in selectionnes]}")


# ============================================================================
# Example 5: U_mem - Cognitive Memory
# ============================================================================


def demo_memoire_cognitive():
    """Demonstrate cognitive memory system."""
    print("\n" + "=" * 70)
    print("Example 5: U_mem - Mémoire Cognitive (Traces & Replay)")
    print("=" * 70)

    memoire = Memoire(nom="Mémoire Centrale")

    print("\n💾 Recording traces...")

    # Record various traces
    memoire.enregistrer(
        type_trace="action",
        contenu={"action": "analyze_data", "duration": 2.5},
        contexte={"user": "alice", "session": "session_1"},
    )

    memoire.enregistrer(
        type_trace="reasoning",
        contenu={"problem": "optimize_query", "steps": 7, "solution": "index_db"},
        contexte={"complexity": "high"},
    )

    memoire.enregistrer(
        type_trace="objective",
        contenu={"name": "improve_performance", "achieved": True},
        contexte={"improvement": "40%"},
    )

    # Query memory
    print(f"\n🔍 Querying memory:")

    # Get all traces
    all_traces = memoire.recuperer()
    print(f"   Total traces: {len(all_traces)}")

    # Get by type
    action_traces = memoire.recuperer(type_trace="action")
    print(f"   Action traces: {len(action_traces)}")

    reasoning_traces = memoire.recuperer(type_trace="reasoning")
    print(f"   Reasoning traces: {len(reasoning_traces)}")

    # Replay a trace
    if reasoning_traces:
        trace_id = reasoning_traces[0].id
        rejouee = memoire.rejouer(trace_id)
        print(f"\n🔄 Replaying trace {trace_id[:8]}...")
        print(f"   Problem: {rejouee['problem']}")
        print(f"   Solution: {rejouee['solution']}")

    # Statistics
    stats = memoire.statistiques()
    print(f"\n📊 Memory Statistics:")
    print(f"   Total: {stats['total_traces']}")
    for type_, count in stats['types'].items():
        print(f"   {type_}: {count}")


# ============================================================================
# Example 6: RRLA - Reflection Reasoning Loop Agent
# ============================================================================


def demo_rrla():
    """Demonstrate reflection reasoning loop."""
    print("\n" + "=" * 70)
    print("Example 6: RRLA - Reflection Reasoning Loop Agent")
    print("=" * 70)

    memoire = Memoire()
    rrla = RRLA(memoire=memoire, enable_logging=False)

    # Perform reasoning
    print("\n🧠 Performing structured reasoning...")

    probleme = "Comment optimiser les performances du système?"
    contexte = {"current_load": "high", "response_time": "slow"}

    cor = rrla.raisonner(probleme=probleme, contexte=contexte, mode="complet")

    print(f"\n📝 Chain of Reasoning (CoR):")
    print(f"   Problem: {cor.probleme}")
    print(f"   Reasoning ID: {cor.id[:8]}...")

    # Show reasoning steps
    etapes = cor.get_etapes()
    print(f"\n🔗 Reasoning Steps ({len(etapes)}):")
    for i, etape in enumerate(etapes, 1):
        print(f"   {i}. {etape.etape.upper()}")
        print(f"      Content: {etape.contenu}")
        print(f"      Confidence: {etape.confiance:.0%}")

    # Get CoR data
    cor_data = cor.to_dict()
    print(f"\n📈 Average Confidence: {cor_data['confiance_moyenne']:.0%}")

    # Verify storage in memory
    reasoning_traces = memoire.recuperer(type_trace="reasoning")
    print(f"\n💾 Reasoning stored in memory: {len(reasoning_traces)} trace(s)")


# ============================================================================
# Example 7: Persona System (Nümtema)
# ============================================================================


def demo_persona():
    """Demonstrate persona system."""
    print("\n" + "=" * 70)
    print("Example 7: Persona - Agent Personality (Nümtema)")
    print("=" * 70)

    # Show Nümtema persona
    print(f"\n🎭 Nümtema Persona:")
    print(f"   Name: {NUMTEMA_PERSONA.nom}")
    print(f"   Description: {NUMTEMA_PERSONA.description}")
    print(f"\n   Traits:")
    for trait in NUMTEMA_PERSONA.traits:
        print(f"     • {trait}")

    print(f"\n   Communication Style: {NUMTEMA_PERSONA.style_communication}")
    print(f"   Detail Level: {NUMTEMA_PERSONA.niveau_detail}")
    print(f"   Advisory Mode: {NUMTEMA_PERSONA.mode_conseil}")

    print(f"\n   Superpowers ({len(NUMTEMA_PERSONA.superpouvoirs)}):")
    for power in NUMTEMA_PERSONA.superpouvoirs:
        print(f"     ⚡ {power}")

    # Create custom persona
    print(f"\n\n🎨 Creating Custom Persona...")

    custom_persona = Persona(
        nom="CodeMaster",
        description="Expert en développement logiciel",
        traits=["précis", "technique", "pédagogue"],
        style_communication="technique",
        niveau_detail="détaillé",
        mode_conseil="directif",
        superpouvoirs=["code_review", "architecture_design", "debugging"],
    )

    print(f"   Created: {custom_persona.nom}")
    print(f"   Traits: {', '.join(custom_persona.traits)}")


# ============================================================================
# Example 8: Complete Workflow Integration
# ============================================================================


def demo_workflow_complet():
    """Demonstrate complete morphic workflow."""
    print("\n" + "=" * 70)
    print("Example 8: Complete Morphic Workflow (All Components)")
    print("=" * 70)

    print("\n🎯 Mission: Train and deploy an AI agent")

    # 1. Setup memory
    memoire = Memoire(nom="System Memory")

    # 2. Create agent (U₀)
    agent = ObjetReactif(nom="Agent_Production", niveau=0)
    print(f"\n1️⃣  Created agent: {agent.nom}")

    # 3. Define objective (U_G)
    def agent_pret_production(obj):
        return (
            obj.niveau >= 10
            and obj.etat == EtatObjet.ACTIF
            and obj.get_donnee("validated") is True
        )

    objectif_production = ObjectifG(
        nom="Agent Ready for Production", condition=agent_pret_production, priorite=10
    )

    print(f"2️⃣  Objective: {objectif_production.nom}")

    # 4. Create flux (U_Φ)
    flux = FluxPhi(nom="Training Pipeline", objectif=objectif_production)

    def phase_training(obj):
        for _ in range(10):
            obj.incrementer_niveau()
        obj.set_donnee("trained", True)
        return obj

    def phase_validation(obj):
        obj.set_donnee("validated", True)
        obj.changer_etat(EtatObjet.ACTIF)
        return obj

    flux.ajouter_etape("Training", phase_training)
    flux.ajouter_etape("Validation", phase_validation)

    print(f"3️⃣  Created flux: {flux.nom}")

    # 5. Execute with reasoning (RRLA)
    rrla = RRLA(memoire=memoire)

    print(f"\n4️⃣  Executing training pipeline...")
    result = flux.executer(agent)

    print(f"   ✅ Objective achieved: {result['objectif_atteint']}")
    print(f"   ✅ Agent level: {agent.niveau}")
    print(f"   ✅ Agent state: {agent.etat.value}")

    # 6. Reason about results
    print(f"\n5️⃣  Reasoning about deployment...")
    cor = rrla.raisonner(
        probleme="Should we deploy this agent to production?",
        contexte={"training_result": result, "agent_level": agent.niveau},
        mode="approfondi",
    )

    print(f"   🧠 Reasoning completed: {len(cor.get_etapes())} steps")
    print(f"   📊 Confidence: {cor.to_dict()['confiance_moyenne']:.0%}")

    # 7. Store in memory
    memoire.enregistrer(
        type_trace="deployment",
        contenu={
            "agent": agent.nom,
            "flux_result": result,
            "reasoning_id": cor.id,
            "decision": "APPROVED",
        },
        contexte={"environment": "production"},
    )

    print(f"\n6️⃣  Stored in memory:")
    stats = memoire.statistiques()
    print(f"   Total traces: {stats['total_traces']}")


# ============================================================================
# Example 9: Flow Integration
# ============================================================================


def demo_flow_integration():
    """Demonstrate Flow integration with MorphicLayer."""
    print("\n" + "=" * 70)
    print("Example 9: Flow Integration - Morphic Agent Pipeline")
    print("=" * 70)

    # Setup components
    def condition(obj):
        return obj.niveau >= 5

    objectif = ObjectifG(nom="Level 5", condition=condition)
    flux = FluxPhi(nom="LevelUp", objectif=objectif)

    def level_up(obj):
        for _ in range(5):
            obj.incrementer_niveau()
        return obj

    flux.ajouter_etape("LevelUp", level_up)

    memoire = Memoire()
    rrla = RRLA(memoire=memoire)

    # Build Flow
    flow = Flow(name="MorphicAgentPipeline")

    # Create custom initialization node
    class InitNode(Node):
        def exec(self, shared: SharedStore) -> dict:
            agent = ObjetReactif(nom="FlowAgent", niveau=0)
            shared.set("objet_reactif", agent)
            shared.set("probleme", "How to level up efficiently?")
            print("   📥 Agent initialized")
            return {"initialized": True}

    # Add nodes
    init_node = InitNode(name="Initialize")
    reasoning_node = ReasoningNode(rrla=rrla, mode="rapide", name="Reason")
    flux_node = FluxExecutionNode(flux=flux, name="ExecuteFlux")
    eval_node = ObjectifEvaluationNode(objectif=objectif, name="EvaluateObjective")
    memory_node = MemoireNode(
        memoire=memoire, type_trace="flow", contenu_key="flux_result", name="StoreMemory"
    )

    flow.add_node(init_node)
    flow.add_node(reasoning_node)
    flow.add_node(flux_node)
    flow.add_node(eval_node)
    flow.add_node(memory_node)

    flow.set_start(init_node)

    # Execute
    print(f"\n🚀 Executing Flow: {flow.name}")
    results = flow.execute()

    print(f"\n✅ Flow Results:")
    print(f"   Objective achieved: {flow._shared.get('objectif_atteint')}")
    print(f"   Memory traces: {len(memoire.recuperer())}")
    print(f"   Nodes executed: {len(results)}")


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "MorphicLayer Demo - Système Morphique Complet" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")

    # Run all examples
    demo_univers_morphique()
    demo_objectifs()
    demo_flux_morphique()
    demo_selecteur_logique()
    demo_memoire_cognitive()
    demo_rrla()
    demo_persona()
    demo_workflow_complet()
    demo_flow_integration()

    print("\n" + "=" * 70)
    print("✨ Demo Complete!")
    print("=" * 70)

    print("\n📚 Summary - Morphic Universe Components:")
    print("\n   U₀: Base Morphic Universe")
    print("     • ObjetReactif: Reactive objects with state and history")
    print("     • Morphisme: Transformations between objects")

    print("\n   U_G: Goal Universe (Postulates G1-G6)")
    print("     • ObjectifG: Boolean goal functions")
    print("     • GenerateurObjectifs: Dynamic objective creation")
    print("     • G1: Objective = Defined Goal ✓")
    print("     • G2: Objective = Boolean Function ✓")
    print("     • G3: Objectives are Evaluable ✓")
    print("     • G4: Objectives Guide Fluxes ✓")
    print("     • G5: Objectives are Memorizable ✓")
    print("     • G6: Objectives are Dynamically Generatable ✓")

    print("\n   U_Φ: Morphic Flux")
    print("     • FluxPhi: Plan generation and execution")
    print("     • Guided by objectives (G4)")

    print("\n   U_Ψ: Logical Selector")
    print("     • SelecteurPsi: Intelligent filtering and routing")

    print("\n   U_mem: Cognitive Memory")
    print("     • Memoire: Trace recording and replay")
    print("     • MemoireVectorielle: Semantic memory")

    print("\n   RRLA: Reflection Reasoning")
    print("     • ChainOfReasoning (CoR): Structured reasoning")
    print("     • RRLA: Multi-step reasoning with reflection")

    print("\n   Persona: Agent Personality")
    print("     • Nümtema: Predefined bienveillant expert")
    print("     • Custom personas with traits and superpowers")

    print("\n   Flow Integration:")
    print("     • ObjectifEvaluationNode, FluxExecutionNode")
    print("     • ReasoningNode, MemoireNode")
    print("     • Seamless integration with existing SDK")

    print("\n🎯 Production-Ready Morphic Agent System!")
    print("=" * 70)
    print("\n")


if __name__ == "__main__":
    main()
