# 🚀 NumAgents CLI Guide Complet

## 📚 Table des Matières

1. [Installation](#installation)
2. [CLI Originale (typer)](#cli-originale)
3. [CLI Enhanced (interactive)](#cli-enhanced)
4. [Commandes Disponibles](#commandes-disponibles)
5. [Exemples d'Utilisation](#exemples-dutilisation)
6. [Configuration](#configuration)

---

## 🔧 Installation

### Prérequis
```bash
cd /home/user/numagents
pip install -e .
pip install rich  # Pour une meilleure expérience visuelle
pip install typer pyyaml  # Pour CLI originale
```

### Ajouter au PATH
```bash
# Ajouter au ~/.bashrc ou ~/.zshrc
export PATH="$PATH:/home/user/numagents/bin"

# Recharger
source ~/.bashrc
```

---

## 📋 CLI Originale (typer)

**Commande**: `num-agents` ou `python -m num_agents.cli`

### Fonctionnalités
- ✅ Génération d'agents à partir de specs YAML
- ✅ Audit d'agents
- ✅ Génération de graphes logiques

### Commandes

#### `generate` - Générer un agent
```bash
num-agents generate agent_spec.yaml

# Options:
--univers-catalog, -u    # Chemin vers catalog universe
--output-dir, -o          # Répertoire de sortie
--skip-graph, -s         # Skip graph generation
--skip-audit, -a         # Skip audit report
```

#### `audit` - Auditer un agent
```bash
num-agents audit ./agent_directory

# Options:
--agent-spec, -a         # Spec YAML
--univers-catalog, -u    # Catalog universe
--output-path, -o        # Chemin rapport audit
```

#### `graph` - Générer graphe logique
```bash
num-agents graph ./agent_directory

# Options:
--output-mermaid, -m     # Sortie Mermaid
--output-markdown, -d    # Sortie Markdown
```

---

## 🎨 CLI Enhanced (Interactive)

**Commande**: `numagent-enhanced` ou `python -m num_agents.cli_enhanced`

### Fonctionnalités
- ✅ Création interactive d'agents
- ✅ Templates prédéfinis (simple, morphic, full)
- ✅ Support complet des 4 layers
- ✅ Interface rich/colorée
- ✅ Wizard interactif

### Commandes

#### `wizard` (ou `w`) - Assistant interactif
```bash
numagent-enhanced wizard
```

**Fonctionnalités**:
1. Choix du nom de l'agent
2. Sélection du template:
   - **simple**: Agent basique minimal
   - **morphic**: Agent avec raisonnement morphique (RRLA)
   - **full**: Agent complet avec les 4 layers
   - **custom**: Choix manuel des layers

3. Configuration interactive des layers (si custom)
4. Génération automatique du code
5. Création de la config YAML

#### `status` (ou `s`) - Statut du système
```bash
numagent-enhanced status
```

**Affiche**:
- Version du SDK
- État des layers (disponibles ou non)
- Configuration actuelle
- Chemins des répertoires

#### `list` (ou `ls`) - Lister les agents
```bash
numagent-enhanced list
```

**Affiche**:
- Tableau de tous les agents créés
- Template utilisé pour chaque agent
- Layers inclus
- Statut de chaque agent

#### `run` (ou `r`) - Exécuter un agent
```bash
numagent-enhanced run <agent_name>

# Options:
--task, -t           # Tâche à exécuter
--data, -d           # Données additionnelles (format JSON)
```

**Exemple**:
```bash
numagent-enhanced run my_agent --task "Process data" --data '{"value": 42}'
```

**Fonctionnalités**:
- Exécute l'agent avec logging automatique
- Supporte agents Flow et agents class-based
- Crée un fichier log dans ~/.numagents/logs
- Affiche le résultat en JSON

#### `inspect` (ou `i`) - Inspector un agent
```bash
numagent-enhanced inspect <agent_name>

# Options:
--show-code, -c      # Afficher aperçu du code
```

**Affiche**:
- Configuration de l'agent
- Template utilisé
- Layers inclus
- Date de création
- Liste des fichiers
- Aperçu du code (si --show-code)

#### `logs` (ou `l`) - Voir les logs
```bash
numagent-enhanced logs <agent_name>

# Options:
--list               # Lister tous les fichiers log
--tail, -n N         # Afficher les N dernières lignes
```

**Exemples**:
```bash
# Voir dernier log
numagent-enhanced logs my_agent

# Lister tous les logs
numagent-enhanced logs my_agent --list

# Voir 50 dernières lignes
numagent-enhanced logs my_agent --tail 50
```

#### `test` (ou `t`) - Tester un agent
```bash
numagent-enhanced test <agent_name>
```

**Fonctionnalités**:
- Exécute scénarios de test automatiques
- Teste avec différentes données
- Affiche résumé (passed/failed)
- Code de sortie: 0 si tous les tests passent, 1 sinon

#### `config` (ou `c`) - Gérer la configuration
```bash
numagent-enhanced config <action>

# Actions:
show    # Afficher configuration
edit    # Éditer configuration (avec $EDITOR)
reset   # Réinitialiser configuration
```

**Exemples**:
```bash
# Voir config
numagent-enhanced config show

# Éditer config
numagent-enhanced config edit

# Reset à défauts
numagent-enhanced config reset
```

---

## 📋 Commandes Disponibles - Résumé

### CLI Originale

| Commande | Description | Exemple |
|----------|-------------|---------|
| `generate` | Générer agent depuis YAML | `num-agents generate spec.yaml` |
| `audit` | Auditer agent existant | `num-agents audit ./my_agent` |
| `graph` | Générer graphe logique | `num-agents graph ./my_agent` |

### CLI Enhanced

| Commande | Alias | Description |
|----------|-------|-------------|
| `wizard` | `w` | Assistant création interactif |
| `status` | `s` | Statut système et layers |
| `list` | `ls` | Lister tous les agents |
| `run` | `r` | Exécuter un agent |
| `inspect` | `i` | Inspector un agent |
| `logs` | `l` | Voir les logs d'un agent |
| `test` | `t` | Tester un agent |
| `config` | `c` | Gérer la configuration |

---

## 💡 Exemples d'Utilisation

### Exemple 1: Créer un agent simple (Enhanced CLI)
```bash
$ numagent-enhanced wizard

╔══════════════════════════════════════════╗
║  🤖  NumAgents SDK - Enhanced CLI  🚀   ║
╚══════════════════════════════════════════╝

Agent name (my_agent): my_first_agent

Available templates:
  1. simple       - Basic agent - minimal setup
  2. morphic      - Morphic agent - goal-oriented with RRLA
  3. full         - Full agent - all 4 layers
  4. custom       - Custom - choose layers

Choose template (1): 1

✅ Agent 'my_first_agent' created!
ℹ️  Location: /home/user/.numagents/agents_enhanced/my_first_agent/my_first_agent.py
ℹ️  Template: simple

📝 Next Steps
1. View code:   cat /home/user/.numagents/agents_enhanced/my_first_agent/my_first_agent.py
2. Edit code:   nano /home/user/.numagents/agents_enhanced/my_first_agent/my_first_agent.py
3. Run agent:   numagent-enhanced run my_first_agent
4. Test agent:  numagent-enhanced test my_first_agent
```

### Exemple 2: Créer un agent morphique
```bash
$ numagent-enhanced wizard

Agent name: reasoning_agent
Choose template: 2 (morphic)

✅ Agent 'reasoning_agent' created with:
- Morphic Universe (U₀, U_G, U_Φ, U_Ψ, U_mem)
- RRLA reasoning
- Nümtema persona
- Cognitive memory
```

### Exemple 3: Agent complet avec tous les layers
```bash
$ numagent-enhanced wizard

Agent name: production_agent
Choose template: 3 (full)

✅ Agent 'production_agent' created with:
- ToolLayer: Action management
- StateLayer: State machines
- SecurityLayer: Auth & sanitization
- MorphicLayer: Goal-oriented reasoning
```

### Exemple 4: Vérifier le statut
```bash
$ numagent-enhanced status

╔══════════════════════════════════════════╗
║  🤖  NumAgents SDK - Enhanced CLI  🚀   ║
╚══════════════════════════════════════════╝

🔧 System Status
═══════════════
✅ SDK Version: 0.1.0

📦 Available Layers
═══════════════════
Layer            │ Status
─────────────────┼───────────────
ToolLayer        │ ✅ Available
StateLayer       │ ✅ Available
SecurityLayer    │ ✅ Available
MorphicLayer     │ ✅ Available

⚙️  Configuration
═══════════════
ℹ️  Agents directory: /home/user/.numagents/agents_enhanced
ℹ️  Logs directory: /home/user/.numagents/logs
ℹ️  Config file: /home/user/.numagents/config_enhanced.yaml
```

### Exemple 5: Lister et inspecter les agents
```bash
# Lister tous les agents
$ numagent-enhanced list

📋 Agents List
=============
Name            | Template | Layers                              | Status
───────────────────────────────────────────────────────────────────────
my_first_agent  | simple   |                                     | ✅ Ready
reasoning_agent | morphic  | morphic                             | ✅ Ready
production_agent| full     | tool, state, security, morphic      | ✅ Ready

Total: 3 agent(s)

# Inspecter un agent
$ numagent-enhanced inspect production_agent

🔍 Inspecting Agent: production_agent
=====================================
ℹ️  Name: production_agent
ℹ️  Template: full
ℹ️  Layers: tool, state, security, morphic
ℹ️  Created: 1729494000.0

📁 Files
========
ℹ️    config.yaml (145 bytes)
ℹ️    production_agent.py (4521 bytes)

# Inspecter avec aperçu code
$ numagent-enhanced inspect production_agent --show-code
```

### Exemple 6: Exécuter et tester des agents
```bash
# Exécuter un agent simple
$ numagent-enhanced run my_first_agent --task "Hello World"

🚀 Running Agent: my_first_agent
===============================
ℹ️  Template: simple
ℹ️  Layers:
ℹ️  Log file: /root/.numagents/logs/my_first_agent_20251021_120000.log

📊 Execution
===========
✅ Execution completed!

📄 Result
========
{
  "my_first_agent": {
    "status": "success",
    "result": "Processed: Hello World"
  }
}

# Exécuter avec données JSON
$ numagent-enhanced run production_agent \
  --task "Process user data" \
  --data '{"user_id": 123, "action": "validate"}'

# Tester un agent
$ numagent-enhanced test my_first_agent

🧪 Testing Agent: my_first_agent
===============================
ℹ️  Running test scenarios...

  Testing: Basic execution
✅     ✅ PASSED

  Testing: With data
✅     ✅ PASSED

📊 Test Summary
==============
ℹ️  Total: 2
✅ Passed: 2
```

### Exemple 7: Voir les logs
```bash
# Voir le dernier log
$ numagent-enhanced logs my_first_agent

📜 Logs for: my_first_agent
==========================
ℹ️  Latest log: my_first_agent_20251021_120000.log

2025-10-21 12:00:00 - INFO - Starting agent execution
2025-10-21 12:00:00 - INFO - Task: Hello World
2025-10-21 12:00:01 - INFO - Execution completed

# Lister tous les logs
$ numagent-enhanced logs my_first_agent --list

📜 Logs for: my_first_agent
==========================
ℹ️  Found 5 log file(s):
  my_first_agent_20251021_120000.log (1024 bytes, 2025-10-21 12:00:00)
  my_first_agent_20251021_110000.log (2048 bytes, 2025-10-21 11:00:00)
  my_first_agent_20251021_100000.log (1536 bytes, 2025-10-21 10:00:00)

# Voir dernières 20 lignes
$ numagent-enhanced logs my_first_agent --tail 20
```

### Exemple 8: Gérer la configuration
```bash
# Afficher configuration
$ numagent-enhanced config show

⚙️  Configuration
================
agents_dir: /root/.numagents/agents_enhanced
default_layers:
  morphic: true
  security: true
  state: true
  tool: true
logs_dir: /root/.numagents/logs
templates:
  simple: Basic agent with minimal setup
  morphic: Goal-oriented agent with RRLA
  full: Full-featured agent with all 4 layers

# Éditer configuration
$ numagent-enhanced config edit
# Opens in $EDITOR (nano, vim, etc.)

# Réinitialiser configuration
$ numagent-enhanced config reset
⚠️  Reset configuration to defaults? (y/N): y
✅ Configuration reset to defaults
```

### Exemple 9: Workflow complet
```bash
# 1. Créer un agent
$ numagent-enhanced wizard
Agent name: data_processor
Choose template: 3 (full)
✅ Agent 'data_processor' created!

# 2. Lister les agents
$ numagent-enhanced list
data_processor | full | tool, state, security, morphic | ✅ Ready

# 3. Inspecter l'agent
$ numagent-enhanced inspect data_processor

# 4. Exécuter l'agent
$ numagent-enhanced run data_processor --task "Process CSV" --data '{"file": "data.csv"}'

# 5. Voir les logs
$ numagent-enhanced logs data_processor --tail 50

# 6. Tester l'agent
$ numagent-enhanced test data_processor
```

### Exemple 10: Générer agent depuis spec YAML (CLI originale)
```bash
# Créer spec
cat > my_agent_spec.yaml <<EOF
name: MyAgent
nodes:
  - name: ProcessNode
    type: custom
flows:
  - name: MainFlow
    start: ProcessNode
EOF

# Générer
num-agents generate my_agent_spec.yaml \
  --output-dir ./my_generated_agent \
  --univers-catalog ./config/univers_catalog.yaml

# Auditer
num-agents audit ./my_generated_agent
```

---

## ⚙️ Configuration

### Fichier de Configuration Enhanced CLI
**Chemin**: `~/.numagents/config_enhanced.yaml`

```yaml
version: 0.2.0
agents_dir: /home/user/.numagents/agents_enhanced
logs_dir: /home/user/.numagents/logs

default_layers:
  tool: true
  state: true
  security: true
  morphic: true

templates:
  simple: Basic agent with minimal setup
  morphic: Goal-oriented agent with RRLA
  full: Full-featured agent with all 4 layers
  conversational: Conversational agent with dialogue management
```

### Personnaliser la Configuration
```bash
# Éditer manuellement
nano ~/.numagents/config_enhanced.yaml

# Ou utiliser la CLI (à venir)
numagent-enhanced config set default_layers.morphic=false
```

---

## 🎯 Choisir la Bonne CLI

### Utilisez **CLI Originale** (`num-agents`) si:
- ✅ Vous avez des specs YAML détaillées
- ✅ Vous voulez générer des graphes logiques
- ✅ Vous voulez des audits automatiques
- ✅ Workflow basé sur YAML

### Utilisez **CLI Enhanced** (`numagent-enhanced`) si:
- ✅ Vous voulez créer rapidement des agents
- ✅ Vous préférez une approche interactive
- ✅ Vous voulez utiliser les templates prédéfinis
- ✅ Vous voulez support complet des 4 layers
- ✅ Workflow interactif et visuel

---

## 🚀 Roadmap CLI

### Phase 1 (Complété) ✅
- ✅ CLI originale (typer)
- ✅ CLI enhanced (interactive)
- ✅ Agent wizard
- ✅ Status command

### Phase 2 (Complété) ✅
- ✅ `run` - Exécuter agents
- ✅ `test` - Tester agents
- ✅ `list` - Lister agents
- ✅ `inspect` - Inspector agents
- ✅ `logs` - Voir logs
- ✅ `config` - Gérer configuration

### Phase 3 (À venir)
- ⏳ `deploy` - Déployer agents
- ⏳ `benchmark` - Benchmarker agents
- ⏳ `monitor` - Monitoring temps réel
- ⏳ `package` - Packager agents
- ⏳ Mode daemon/serveur
- ⏳ Multi-agent orchestration commands

---

## 📚 Ressources

- **Documentation**: `/home/user/numagents/docs/`
- **Exemples**: `/home/user/numagents/examples/`
- **Tests**: `/home/user/numagents/tests/`
- **Config**: `~/.numagents/`

---

## 🆘 Support

### Problèmes Courants

**Erreur: Command not found**
```bash
# Solution: Ajouter au PATH
export PATH="$PATH:/home/user/numagents/bin"
```

**Erreur: Module not found**
```bash
# Solution: Installer le SDK
cd /home/user/numagents
pip install -e .
```

**Erreur: Rich not available**
```bash
# Solution: Installer rich (optionnel)
pip install rich
```

---

## 🎊 Conclusion

Vous avez maintenant **2 CLI puissantes** :

1. **CLI Originale** - Génération depuis YAML, audits, graphes
2. **CLI Enhanced** - Interactive, templates, wizard

**Combinées** = Workflow complet pour développement d'agents IA! 🚀
