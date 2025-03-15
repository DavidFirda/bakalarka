# bakalarka

postup pri dátach: 
1. rawdata_analysis
2. data_categorization
3. data_analysis
4. generation_incorrect_output

algoritmus:
1. Q-learning
2. POMDP
3. POMDP-v2

# Porovnanie Politík v TutorPOMDP

Dve rôzne politiky použité v modeloch **TutorPOMDP**.

| **Parameter** | **Druhý model (ε-greedy + streak correction)** | **Prvý model (softmax-based)** |
|--------------|--------------------------------|---------------------------------|
| **Výber akcie** | ε-greedy, pričom preferuje slabé kategórie | Softmax na základe neistoty a slabých kategórií |
| **Slabé kategórie** | Preferuje, ale pri sérii chýb prepne na silnú kategóriu | Extra váha pri výbere na základe neistoty |
| **Náhodnosť** | Riadená **ε-greedy** stratégiou | Ovládaná **softmaxom** s teplotným parameterom |
| **Dynamické učenie** | Áno, cez prepínanie medzi slabými a silnými kategóriami | Áno, cez zmenu váh a entropiu |
| **Reakcia na sériu nesprávnych odpovedí** |  Prepína na silnú kategóriu pri 3 nesprávnych odpovediach | Nezohľadňuje explicitne streaky |
| **Matematická sofistikovanosť** | Nižšia (pravidlá + ε-greedy) | Vyššia (softmax, entropia) |

## **Zhrnutie**

- **Prvý model (ε-greedy + streak correction)** je jednoduchší a intuitívnejší, pričom obsahuje mechanizmus na posilnenie sebavedomia študenta pri sérii nesprávnych odpovedí.
- **Druhý model (softmax-based)** je sofistikovanejší, dynamicky sa prispôsobuje študentovi pomocou entropie a pravdepodobností.


