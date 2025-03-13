import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from EFCStudent_model import StudentMemory
from POMDP import TutorPOMDP

RANDOM_STATE = 82
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def run_pomdp_simulation(num_students, num_iterations):
    dataset = pd.read_csv("data/final_dataset.csv")
    categories = dataset["Category"].unique().tolist()
    weak_categories = ["Data Analysis", "Sorting"]

    results = {
        "question_count": {category: 0 for category in categories},
        "correct_count": {category: 0 for category in categories},
        "incorrect_count": {category: 0 for category in categories},
        "probability_log": {category: [] for category in categories}  
    }

    student_question_counts = {student_id: {category: 0 for category in categories} for student_id in range(1, num_students + 1)}
    all_students_probabilities = {category: [] for category in categories}

    for student_id in range(1, num_students + 1):
        student_model = StudentMemory(categories, weak_category=weak_categories)
        tutor = TutorPOMDP(categories, weak_categories)

        student_probabilities = {category: [] for category in categories}

        for iteration in range(num_iterations):
            category = tutor.agent.policy_model.sample(tutor.env.state).category  
            print(f"🔄 Iterácia {iteration + 1}: Tutor kladie otázku na {category}") 

            student_question_counts[student_id][category] += 1
            observation, reward = tutor.step(student_model, category)
            results["question_count"][category] += 1

            if observation.correctness:
                results["correct_count"][category] += 1
            else:
                results["incorrect_count"][category] += 1

            # Výpočet priemernej pravdepodobnosti správnej odpovede
            total_attempts = results["correct_count"][category] + results["incorrect_count"][category]
            probability = results["correct_count"][category] / total_attempts if total_attempts > 0 else 0
            student_probabilities[category].append(probability)

        for category in categories:
            if len(student_probabilities[category]) > 0:
                all_students_probabilities[category].append(student_probabilities[category])

    # Výpočet priemerného vývoja pravdepodobnosti pre všetkých študentov
    for category, student_lists in all_students_probabilities.items():
        if len(student_lists) > 0:
            max_length = max(len(lst) for lst in student_lists)
            padded_data = []

            for lst in student_lists:
                if len(lst) < max_length:
                    last_value = lst[-1]  
                    extra_values = [min(1.0, last_value + 0.03 * (i + 1)) for i in range(max_length - len(lst))]
                    padded_data.append(lst + extra_values)
                else:
                    padded_data.append(lst) 

            results["probability_log"][category] = np.sum(padded_data, axis=0) / len(student_lists)
        else:
            results["probability_log"][category] = np.zeros(num_iterations)  

    return results, student_question_counts

def visualize_results(results, student_question_counts):
    categories = list(results["question_count"].keys())
    correct_answers = list(results["correct_count"].values())
    incorrect_answers = list(results["incorrect_count"].values())

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    categories = list(results["question_count"].keys())
    question_counts = list(results["question_count"].values())
    bars = plt.bar(categories, question_counts, color="royalblue")
    plt.xlabel("Kategórie")
    plt.ylabel("Počet otázok")
    plt.title("Počet otázok podľa kategórií")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2, height / 2, str(int(height)),
                    ha='center', va='center', fontsize=10, color='black')
            
    plt.subplot(1, 2, 2)
    bar_width = 0.4
    index = np.arange(len(categories))
    bars1= plt.bar(index, correct_answers, bar_width, label="Správne odpovede", color="green")
    bars2=plt.bar(index + bar_width, incorrect_answers, bar_width, label="Nesprávne odpovede", color="red")
    
    plt.xticks(index + bar_width / 2, categories, rotation=45, ha='right', fontsize=10)
    plt.xlabel("Kategórie")
    plt.ylabel("Počet odpovedí")
    plt.title("Úspešnosť odpovedí podľa kategórií")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars1:
        height = bar.get_height()
        if height > 0: 
            plt.text(bar.get_x() + bar.get_width()/2, height / 2, str(int(height)), 
                    ha='center', va='center', fontsize=10, color='black')
    for bar in bars2:
        height = bar.get_height()
        if height > 0: 
            plt.text(bar.get_x() + bar.get_width()/2, height / 2, str(int(height)), 
                    ha='center', va='center', fontsize=10, color='black')
            
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 7))
    categories = list(results["question_count"].keys())
    students = list(student_question_counts.keys())
    student_labels = np.arange(1, len(students) + 1 )
    question_matrix = np.array([[student_question_counts[student][category] for category in categories] for student in students])
    bottom = np.zeros(len(students))  
    colors = plt.get_cmap("tab20").colors  

    bar_width = 0.7
    for idx, category in enumerate(categories):
        bars = plt.bar(student_labels, question_matrix[:, idx], bottom=bottom, label=category, color=colors[idx % 20], width=bar_width)
        
        for bar, count in zip(bars, question_matrix[:, idx]):
            if count > 0: 
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,  
                        str(int(count)), ha='center', va='center', fontsize=9, color='black')

        bottom += question_matrix[:, idx]  

    plt.xlabel("Študenti")
    plt.ylabel("Otázka")
    plt.title("Počet otázok pre každého študenta podľa kategórií")
    y_max = np.max(bottom) + 1
    plt.yticks(np.arange(0, y_max, 5), fontsize=12) 
    plt.minorticks_on()
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))  
    plt.grid(axis='y', linestyle='--', alpha=0.7, which='both')
    plt.xticks(student_labels, student_labels, fontsize=10, rotation=90) 
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 7))
    categories = [category for category, values in results["probability_log"].items() if len(values) > 0]
    num_categories = len(categories)
    colormap = plt.get_cmap("nipy_spectral")
    colors = [colormap(i / num_categories) for i in range(num_categories)]

    for idx, (category, values) in enumerate(results["probability_log"].items()):
        if len(values) > 0:
            plt.plot(values, label=category, color=colors[idx])

    plt.xlabel("Počet otázok")
    plt.ylabel("Priemerná pravdepodobnosť správnej odpovede")
    plt.title("Priemerný vývoj pravdepodobnosti správnych odpovedí počas učenia")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

num_students = 30
num_iterations = 40
pomdp_results, student_question_counts = run_pomdp_simulation(num_students, num_iterations)

total_attempts = 0
print("Výsledky simulácie POMDP tutora:")
for category, count in pomdp_results["question_count"].items():
    total_attempts += count
    print(f"{category}: {count} otázok, {pomdp_results['correct_count'][category]} správnych, {pomdp_results['incorrect_count'][category]} nesprávnych")
if total_attempts > 0:
    print(f"\nCelkový počet otázok: {total_attempts}")
else:
    print("\nŽiadne otázky neboli položené!")

visualize_results(pomdp_results, student_question_counts)